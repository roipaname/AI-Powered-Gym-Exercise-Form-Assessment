"""
pipeline/workers.py
Top-level worker functions for multiprocessing.
Must be defined at module level (not nested) for pickle compatibility.

Key fix for 0-reps issue:
  - MediaPipe confidence thresholds lowered to 0.4/0.4
  - Visibility filter relaxed to 0.3 (gym videos often have partial occlusion)
  - Frame skip applied AFTER reading, not as a cap on reading
  - FPS read from video metadata for accurate timestamps
"""

import os
import sys
from pathlib import Path


def _setup_worker():
    """Suppress all MediaPipe / TF logging inside worker processes."""
    os.environ["GLOG_minloglevel"]       = "3"
    os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
    os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
    import warnings
    warnings.filterwarnings("ignore")
    # Redirect stderr to /dev/null to kill GL/inference_feedback spam
    try:
        devnull = open(os.devnull, 'w')
        sys.stderr = devnull
    except Exception:
        pass


MIN_REP_FRAMES = 2


# ══════════════════════════════════════════════════════════════════════════════
# TEMPORAL WORKER  (Squat)
# ══════════════════════════════════════════════════════════════════════════════

def _process_clip_temporal(args: tuple) -> list:
    """
    Worker for Squat temporal labels.

    clip_key = video stem = "VIDEOID_PERSONID" (e.g. "45823_6")
    For each rep detected: compute midpoint timestamp, check error intervals.

    Args:
        vpath:          path to video file
        clip_intervals: {error_name: [(start, end), ...]}  for this clip
        split_name:     "train" / "val" / "test"
        frame_skip:     process every Nth frame
        frame_width:    resize width before MediaPipe
        model_complexity: 0/1/2

    Returns:
        list of (fvec, clip_key, split_name, {error: 0/1})
    """
    _setup_worker()

    import cv2
    import numpy as np
    from modules.extractor import PoseExtractor
    from modules.segmentor import RepSegmentor
    from modules.features  import compute_rep_features, feature_vector
    from pipeline.helpers  import timestamp_in_intervals

    (vpath, clip_intervals, split_name,
     frame_skip, frame_width, model_complexity) = args

    clip_key  = Path(vpath).stem
    rows      = []

    cap = cv2.VideoCapture(str(vpath))
    if not cap.isOpened():
        return rows

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    # Relaxed thresholds — gym footage is often occluded / side-on
    extractor = PoseExtractor(
        min_detection_conf=0.4,
        min_tracking_conf=0.4,
        model_complexity=model_complexity,
    )
    segmentor = RepSegmentor()

    lm_buf = []   # landmark arrays for kept frames
    ts_buf = []   # original-video timestamps for kept frames (seconds)
    fi     = 0    # original frame index

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if fi % frame_skip == 0:
            h, w = frame.shape[:2]
            if w > frame_width:
                frame = cv2.resize(
                    frame, (frame_width, int(h * frame_width / w)),
                    interpolation=cv2.INTER_LINEAR)
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            lm, _ = extractor.extract(rgb)
            if lm is not None:
                segmentor.update(lm)
                lm_buf.append(lm)
                ts_buf.append(fi / fps)   # timestamp uses original frame index
        fi += 1

    cap.release()
    extractor.close()

    if not lm_buf:
        return rows

    for rep_obj in segmentor.get_reps():
        s_idx = rep_obj.start_frame
        e_idx = min(rep_obj.end_frame, len(lm_buf) - 1)
        seq   = lm_buf[s_idx: e_idx + 1]
        if len(seq) < MIN_REP_FRAMES:
            continue

        mid_idx = min((s_idx + e_idx) // 2, len(ts_buf) - 1)
        mid_ts  = ts_buf[mid_idx]

        try:
            fvec = feature_vector(compute_rep_features(seq))
        except Exception:
            continue

        label_map = {
            err: int(timestamp_in_intervals(mid_ts, intervals))
            for err, intervals in clip_intervals.items()
        }
        rows.append((fvec, clip_key, split_name, label_map))

    return rows


# ══════════════════════════════════════════════════════════════════════════════
# PER-REP WORKER  (OHP)
# ══════════════════════════════════════════════════════════════════════════════

def _process_clip_per_rep(args: tuple) -> list:
    """
    Worker for OHP per-rep binary labels.
    clip_key = "VIDEOID_PERSONID_REPIDX", matched via label_index.
    """
    _setup_worker()

    import cv2
    import numpy as np
    from modules.extractor import PoseExtractor
    from modules.segmentor import RepSegmentor
    from modules.features  import compute_rep_features, feature_vector

    (vpath, label_index, key_to_split,
     frame_skip, frame_width, model_complexity) = args

    base_id   = Path(vpath).stem.split("_")[0]
    rows      = []

    cap = cv2.VideoCapture(str(vpath))
    if not cap.isOpened():
        return rows

    extractor = PoseExtractor(
        min_detection_conf=0.4,
        min_tracking_conf=0.4,
        model_complexity=model_complexity,
    )
    segmentor = RepSegmentor()

    lm_buf = []
    fi = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if fi % frame_skip == 0:
            h, w = frame.shape[:2]
            if w > frame_width:
                frame = cv2.resize(
                    frame, (frame_width, int(h * frame_width / w)),
                    interpolation=cv2.INTER_LINEAR)
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            lm, _ = extractor.extract(rgb)
            if lm is not None:
                segmentor.update(lm)
                lm_buf.append(lm)
        fi += 1

    cap.release()
    extractor.close()

    vid_index = label_index.get(base_id, {})
    for rep_obj in segmentor.get_reps():
        rep_idx  = rep_obj.rep_number - 1
        rep_data = vid_index.get(rep_idx)
        if not rep_data:
            continue
        found_key = rep_data.get("_key")
        if not found_key:
            continue
        e_idx = min(rep_obj.end_frame, len(lm_buf) - 1)
        seq   = lm_buf[rep_obj.start_frame: e_idx + 1]
        if len(seq) < MIN_REP_FRAMES:
            continue
        try:
            fvec = feature_vector(compute_rep_features(seq))
        except Exception:
            continue
        label_map  = {k: v for k, v in rep_data.items() if k != "_key"}
        split_name = key_to_split.get(found_key, "train")
        rows.append((fvec, found_key, split_name, label_map))

    return rows


# ══════════════════════════════════════════════════════════════════════════════
# IMAGE WORKER  (BarbellRow / ShallowSquat)
# ══════════════════════════════════════════════════════════════════════════════

def _process_image(args: tuple):
    """
    Worker for image-based datasets.
    Returns (fvec, key, split, label_map) or None.
    """
    _setup_worker()

    import cv2
    from modules.extractor import PoseExtractor
    from modules.features  import feature_vector
    from pipeline.helpers  import single_frame_features

    img_path, label_dicts, key_to_split, frame_width, model_complexity = args

    key = Path(img_path).stem
    if not any(key in ld for ld in label_dicts.values()):
        return None

    img = cv2.imread(str(img_path))
    if img is None:
        return None

    h, w = img.shape[:2]
    if w > frame_width:
        img = cv2.resize(img, (frame_width, int(h * frame_width / w)),
                         interpolation=cv2.INTER_LINEAR)

    extractor = PoseExtractor(
        min_detection_conf=0.4,
        min_tracking_conf=0.4,
        model_complexity=model_complexity,
    )
    lm, _ = extractor.extract(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    extractor.close()

    if lm is None:
        return None

    fvec      = feature_vector(single_frame_features(lm))
    label_map = {e: ld.get(key, 0) for e, ld in label_dicts.items()}
    return (fvec, key, key_to_split.get(key, "train"), label_map)
