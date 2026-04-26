"""
train.py — Fitness-AQA Training Pipeline  (Optimized)
======================================================

Speed optimisations applied:
  1. Frame skipping       — process every Nth frame (default N=3) → 3x faster
  2. Frame resizing       — downscale to 640px wide before MediaPipe → 1.5-2x faster
  3. model_complexity=0   — lighter MediaPipe model → 1.5x faster
  4. Multiprocessing      — parallel clip processing across CPU cores → 4-8x faster
  5. Pre-built label index — O(1) key lookup instead of O(n) scan per rep
  6. Image threading      — image datasets processed in parallel with ThreadPool

Dataset layout handled:

data/
├── Squat/
│   ├── Labeled_Dataset/
│   │   ├── splits/  train_keys.json, val_keys.json, test_keys.json, traj_nan.json(ignored)
│   │   └── labels/  error_knees_forward.json, error_knees_inward.json
│   │                Shallow_Squat_Error_Dataset/
│   │                  images/, splits/(train_ids/val_ids/test_ids.json),
│   │                  labels_shallow_depth.json
│   └── videos/  (extract videos.zip here)
├── OHP/
│   ├── Labeled_Dataset/
│   │   ├── splits/  train_keys.json, val_keys.json, test_keys.json
│   │   └── labels/  error_elbows.json, error_knees.json
│   └── videos/
└── BarbellRow/
    ├── Labeled_Dataset/
    │   ├── Splits/  Splits_Lumbar_error/, Splits_TorsoAngle_error/
    │   └── labels/  labels_lumbar_error.json, labels_torso_angle_error.json
    └── images/  (extract BarbellRow_images.zip here)

Usage:
  python train.py --exercise all
  python train.py --exercise Squat --max_samples 100
  python train.py --exercise BarbellRow --max_samples 500
  python train.py --exercise all --dry_run
  python train.py --exercise Squat --workers 4 --frame_skip 2
"""

import argparse
import json
import cv2
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor
import time

from modules.features   import compute_rep_features, feature_vector, FEATURE_KEYS
from modules.classifier import FormClassifier


# ══════════════════════════════════════════════════════════════════════════════
# SPEED DEFAULTS  — override via CLI flags
# ══════════════════════════════════════════════════════════════════════════════

FRAME_SKIP       = 3      # process every Nth frame  (3 = ~10fps from 30fps source)
FRAME_WIDTH      = 640    # resize width before MediaPipe
MODEL_COMPLEXITY = 0      # 0=fast, 1=balanced, 2=accurate
MIN_REP_FRAMES   = 2      # discard reps shorter than this


# ══════════════════════════════════════════════════════════════════════════════
# EXERCISE CONFIG
# ══════════════════════════════════════════════════════════════════════════════

EXERCISE_CONFIG = {

    "Squat": {
        "clf_name":          "BackSquat",
        "data_mode":         "video",
        "video_dir":         "videos",
        "splits_dir":        "Labeled_Dataset/splits",
        "split_files":       {"train": "train_keys.json",
                              "val":   "val_keys.json",
                              "test":  "test_keys.json"},
        "splits_ignore":     {"traj_nan.json"},
        "labels_dir":        "Labeled_Dataset/labels",
        "label_files":       {"knees_forward": "error_knees_forward.json",
                              "knees_inward":  "error_knees_inward.json"},
        "shallow_squat_dir": "Labeled_Dataset/labels/Shallow_Squat_Error_Dataset",
        "per_error_splits":  None,
    },

    "OHP": {
        "clf_name":          "OverheadPress",
        "data_mode":         "video",
        "video_dir":         "videos",
        "splits_dir":        "Labeled_Dataset/splits",
        "split_files":       {"train": "train_keys.json",
                              "val":   "val_keys.json",
                              "test":  "test_keys.json"},
        "splits_ignore":     set(),
        "labels_dir":        "Labeled_Dataset/labels",
        "label_files":       {"elbow_error": "error_elbows.json",
                              "knees_error": "error_knees.json"},
        "shallow_squat_dir": None,
        "per_error_splits":  None,
    },

    "BarbellRow": {
        "clf_name":          "BarbellRow",
        "data_mode":         "image",
        "image_dir":         "images",
        "splits_dir":        None,
        "split_files":       None,
        "labels_dir":        "Labeled_Dataset/labels",
        "label_files":       {"lumbar_error":      "labels_lumbar_error.json",
                              "torso_angle_error": "labels_torso_angle_error.json"},
        "shallow_squat_dir": None,
        "per_error_splits": {
            "lumbar_error": {
                "splits_dir":  "Labeled_Dataset/Splits/Splits_Lumbar_error",
                "split_files": {"train": "train_Ids.json",
                                "val":   "val_Ids.json",
                                "test":  "test_Ids.json"},
            },
            "torso_angle_error": {
                "splits_dir":  "Labeled_Dataset/Splits/Splits_TorsoAngle_error",
                "split_files": {"train": "train_Ids.json",
                                "val":   "val_Ids.json",
                                "test":  "test_Ids.json"},
            },
        },
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def load_json(path: Path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def parse_label_value(v):
    """Safely convert any label value format to 0/1 int or None."""
    if isinstance(v, (int, float)):
        return int(v)
    if isinstance(v, list) and v and isinstance(v[0], (int, float)):
        return int(v[0])
    if isinstance(v, dict):
        nums = [x for x in v.values() if isinstance(x, (int, float))]
        if nums:
            return int(nums[0])
    return None


def load_labels(labels_dir: Path, label_files: dict) -> dict:
    """Returns labels[error_name][clip_key] = 0 or 1"""
    labels = {}
    for error_name, fname in label_files.items():
        fpath = labels_dir / fname
        data  = load_json(fpath)
        if data is None:
            print(f"    [warn] Missing: {fpath}")
            continue
        parsed = {}
        skipped = 0
        for k, v in data.items():
            val = parse_label_value(v)
            if val is not None:
                parsed[str(k)] = val
            else:
                skipped += 1
        if skipped:
            print(f"    [warn] {error_name}: skipped {skipped} unreadable entries")
        if not parsed:
            print(f"    [warn] {error_name}: no valid entries — skipping")
            continue
        labels[error_name] = parsed
        pos   = sum(parsed.values())
        total = len(parsed)
        print(f"    {error_name}: {total:,} entries | {pos} positive ({100*pos/total:.1f}%)")
    return labels


def load_split_keys(splits_dir: Path, split_files: dict) -> dict:
    splits = {}
    for split_name, fname in split_files.items():
        fpath = splits_dir / fname
        data  = load_json(fpath)
        if data is None:
            print(f"    [warn] Missing split: {fpath}")
            splits[split_name] = set()
        else:
            splits[split_name] = set(str(k) for k in data)
            print(f"    split '{split_name}': {len(splits[split_name]):,} keys")
    return splits


def build_label_index(all_labels: dict) -> dict:
    """
    Pre-build O(1) lookup: video_id -> {rep_idx -> {error: label, _key: str}}
    Eliminates O(n) label scan per rep in the video loop.
    """
    index = {}
    for error_name, lbl_dict in all_labels.items():
        for key, label in lbl_dict.items():
            parts = key.split("_")
            if len(parts) != 3:
                continue
            vid_id  = parts[0]
            rep_idx = int(parts[2])
            if vid_id not in index:
                index[vid_id] = {}
            if rep_idx not in index[vid_id]:
                index[vid_id][rep_idx] = {"_key": key}
            index[vid_id][rep_idx][error_name] = label
    return index


# ══════════════════════════════════════════════════════════════════════════════
# SINGLE-FRAME FEATURES  (image datasets)
# ══════════════════════════════════════════════════════════════════════════════

def _angle(a, b, c) -> float:
    ba = a - b; bc = c - b
    n1 = np.linalg.norm(ba); n2 = np.linalg.norm(bc)
    if n1 < 1e-8 or n2 < 1e-8:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(ba, bc) / (n1 * n2), -1, 1))))


def single_frame_features(lm: np.ndarray) -> dict:
    xyz = lm[:, :3]
    kL = _angle(xyz[23], xyz[25], xyz[27]); kR = _angle(xyz[24], xyz[26], xyz[28])
    hL = _angle(xyz[11], xyz[23], xyz[25]); hR = _angle(xyz[12], xyz[24], xyz[26])
    eL = _angle(xyz[11], xyz[13], xyz[15]); eR = _angle(xyz[12], xyz[14], xyz[16])
    sp = _angle(xyz[11], xyz[23], xyz[25])
    return {
        "knee_angle_min":    min(kL, kR),   "knee_angle_max":    max(kL, kR),
        "knee_angle_range":  abs(kL - kR),  "hip_angle_min":     min(hL, hR),
        "hip_angle_max":     max(hL, hR),   "elbow_angle_min":   min(eL, eR),
        "elbow_angle_max":   max(eL, eR),   "spine_angle_min":   sp,
        "spine_angle_mean":  sp,            "knee_symmetry":     abs(kL - kR),
        "hip_symmetry":      abs(hL - hR),  "elbow_symmetry":    abs(eL - eR),
        "depth_range":       float(abs(lm[23, 1] - lm[24, 1])),
        "knee_temporal_var": 0.0,           "hip_temporal_var":  0.0,
        "knee_avg_velocity": 0.0,
    }


# ══════════════════════════════════════════════════════════════════════════════
# WORKER FUNCTIONS  (top-level — required for multiprocessing pickling)
# ══════════════════════════════════════════════════════════════════════════════

def _process_clip(args: tuple) -> list:
    """
    Worker: process one video clip end-to-end.
    Returns list of (fvec, key, split, label_map) tuples.
    """
    vpath, label_index, key_to_split, frame_skip, frame_width, model_complexity = args

    import cv2
    import numpy as np
    from modules.extractor import PoseExtractor
    from modules.segmentor import RepSegmentor
    from modules.features  import compute_rep_features, feature_vector

    extractor = PoseExtractor(min_detection_conf=0.5, min_tracking_conf=0.5,
                              model_complexity=model_complexity)
    segmentor = RepSegmentor()
    base_id   = Path(vpath).stem.split("_")[0]
    rows      = []

    cap = cv2.VideoCapture(str(vpath))
    if not cap.isOpened():
        extractor.close()
        return rows

    lm_buf = []
    fi     = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if fi % frame_skip == 0:
            h, w = frame.shape[:2]
            if w > frame_width:
                frame = cv2.resize(frame, (frame_width, int(h * frame_width / w)),
                                   interpolation=cv2.INTER_LINEAR)
            rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
        seq = lm_buf[rep_obj.start_frame: rep_obj.end_frame + 1]
        if len(seq) < 2:
            continue
        try:
            fvec = feature_vector(compute_rep_features(seq))
        except Exception:
            continue
        label_map  = {k: v for k, v in rep_data.items() if k != "_key"}
        split_name = key_to_split.get(found_key, "train")
        rows.append((fvec, found_key, split_name, label_map))

    return rows


def _process_image(args: tuple):
    """
    Worker: process one image file.
    Returns (fvec, key, split, label_map) or None.
    """
    img_path, label_dicts, key_to_split, frame_width, model_complexity = args

    import cv2
    import numpy as np
    from modules.extractor import PoseExtractor

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

    extractor = PoseExtractor(min_detection_conf=0.5, min_tracking_conf=0.5,
                              model_complexity=model_complexity)
    lm, _ = extractor.extract(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    extractor.close()

    if lm is None:
        return None

    fvec      = feature_vector(single_frame_features(lm))
    label_map = {e: ld.get(key, 0) for e, ld in label_dicts.items()}
    return (fvec, key, key_to_split.get(key, "train"), label_map)


# ══════════════════════════════════════════════════════════════════════════════
# IMAGE DATASET BUILDER  (BarbellRow)
# ══════════════════════════════════════════════════════════════════════════════

def build_image_dataset(ex_dir: Path, cfg: dict,
                        max_samples: int = None, dry_run: bool = False,
                        n_workers: int = 4, frame_width: int = FRAME_WIDTH,
                        model_complexity: int = MODEL_COMPLEXITY):

    image_dir  = ex_dir / cfg["image_dir"]
    labels_dir = ex_dir / cfg["labels_dir"]

    if not image_dir.exists():
        print(f"  [error] Not found: {image_dir}  (extract images zip there)")
        return None, None, None, None

    print(f"\n  Loading labels ...")
    all_labels = load_labels(labels_dir, cfg["label_files"])
    if not all_labels:
        return None, None, None, None

    print(f"\n  Loading splits ...")
    key_to_split = {}
    for error_name, split_cfg in cfg["per_error_splits"].items():
        s_dir  = ex_dir / split_cfg["splits_dir"]
        splits = load_split_keys(s_dir, split_cfg["split_files"])
        for sname, keyset in splits.items():
            for k in keyset:
                if k not in key_to_split:
                    key_to_split[k] = sname

    images = (sorted(image_dir.glob("*.jpg")) +
              sorted(image_dir.glob("*.jpeg")) +
              sorted(image_dir.glob("*.png")))
    print(f"\n  Found {len(images):,} images")
    if max_samples:
        images = images[:max_samples]
    if dry_run:
        print("  [DRY RUN] skipping.")
        return None, None, None, None

    args_list = [(str(p), all_labels, key_to_split, frame_width, model_complexity)
                 for p in images]

    X_rows, y_rows, keys_out, splits_out = [], {e: [] for e in all_labels}, [], []
    no_pose = 0
    t0 = time.time()

    print(f"  Processing on {n_workers} threads ...")
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        for i, result in enumerate(pool.map(_process_image, args_list)):
            if (i + 1) % 500 == 0:
                eta = (time.time() - t0) / (i + 1) * (len(images) - i - 1)
                print(f"  [{i+1}/{len(images)}] ETA {eta/60:.1f} min", flush=True)
            if result is None:
                no_pose += 1
                continue
            fvec, key, split, label_map = result
            X_rows.append(fvec)
            keys_out.append(key)
            splits_out.append(split)
            for e in all_labels:
                y_rows[e].append(label_map.get(e, 0))

    print(f"\n  Collected: {len(X_rows):,} | No pose: {no_pose:,} | "
          f"Time: {(time.time()-t0)/60:.1f} min")
    return (np.array(X_rows, dtype=np.float32),
            {e: np.array(v, dtype=int) for e, v in y_rows.items()},
            np.array(splits_out), keys_out)


# ══════════════════════════════════════════════════════════════════════════════
# SHALLOW SQUAT LOADER
# ══════════════════════════════════════════════════════════════════════════════

def load_shallow_squat_dataset(ex_dir: Path, shallow_dir_rel: str,
                               max_samples: int = None, dry_run: bool = False,
                               n_workers: int = 4, frame_width: int = FRAME_WIDTH,
                               model_complexity: int = MODEL_COMPLEXITY):
    s_dir = ex_dir / shallow_dir_rel
    if not s_dir.exists():
        print(f"    [warn] Not found: {s_dir} — skipping shallow squat.")
        return [], {}, [], []

    # Find label JSON
    label_dict = {}
    for fpath in [f for f in s_dir.glob("*.json") if "readme" not in f.name.lower()]:
        data = load_json(fpath)
        if isinstance(data, dict):
            for k, v in data.items():
                val = parse_label_value(v)
                if val is not None:
                    label_dict[str(k)] = val
            pos = sum(label_dict.values())
            print(f"    shallow_squat ({fpath.name}): {len(label_dict):,} | {pos} positive")
    if not label_dict:
        print(f"    [warn] No label data found in {s_dir}")
        return [], {}, [], []

    # Load splits
    key_to_split = {}
    splits_dir   = s_dir / "splits"
    if splits_dir.exists():
        for sname, fname in [("train","train_ids.json"),("val","val_ids.json"),("test","test_ids.json")]:
            data = load_json(splits_dir / fname)
            if data:
                for k in data:
                    key_to_split[str(k)] = sname
                print(f"    shallow split '{sname}': {len(data):,}")

    img_dir = s_dir / "images"
    if not img_dir.exists():
        print(f"    [warn] No images/ in {s_dir} — extract the zip there.")
        return [], {}, [], []

    images = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    print(f"    shallow_squat: {len(images):,} images")
    if max_samples:
        images = images[:max_samples]
    if dry_run:
        return [], {}, [], []

    label_dicts = {"shallow_squat": label_dict}
    args_list   = [(str(p), label_dicts, key_to_split, frame_width, model_complexity)
                   for p in images]

    X_rows, y_ss, keys_out, splits_out = [], [], [], []
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        for result in pool.map(_process_image, args_list):
            if result is None:
                continue
            fvec, key, split, label_map = result
            X_rows.append(fvec)
            keys_out.append(key)
            splits_out.append(split)
            y_ss.append(label_map.get("shallow_squat", 0))

    print(f"    shallow_squat collected: {len(X_rows):,}")
    return X_rows, {"shallow_squat": y_ss}, splits_out, keys_out


# ══════════════════════════════════════════════════════════════════════════════
# VIDEO DATASET BUILDER  (Squat / OHP) — multiprocessing
# ══════════════════════════════════════════════════════════════════════════════

def build_video_dataset(ex_dir: Path, cfg: dict,
                        max_clips: int = None, dry_run: bool = False,
                        n_workers: int = None, frame_skip: int = FRAME_SKIP,
                        frame_width: int = FRAME_WIDTH,
                        model_complexity: int = MODEL_COMPLEXITY):

    video_dir  = ex_dir / cfg["video_dir"]
    labels_dir = ex_dir / cfg["labels_dir"]
    splits_dir = ex_dir / cfg["splits_dir"]

    if not video_dir.exists():
        print(f"  [error] Not found: {video_dir}  (extract videos.zip there)")
        return None, None, None, None

    print(f"\n  Loading labels ...")
    all_labels = load_labels(labels_dir, cfg["label_files"])
    if not all_labels and not cfg.get("shallow_squat_dir"):
        print("  [error] No labels loaded.")
        return None, None, None, None

    print(f"\n  Loading splits ...")
    ignore       = cfg.get("splits_ignore", set())
    active_splits = {k: v for k, v in cfg["split_files"].items() if v not in ignore}
    splits        = load_split_keys(splits_dir, active_splits)
    key_to_split  = {}
    for sname, keyset in splits.items():
        for k in keyset:
            key_to_split[k] = sname

    print(f"\n  Building label index ...")
    label_index = build_label_index(all_labels)
    print(f"  Index: {len(label_index):,} video IDs")

    videos = sorted(video_dir.glob("*.mp4")) + sorted(video_dir.glob("*.avi"))
    print(f"\n  Found {len(videos):,} video files")
    if max_clips:
        videos = videos[:max_clips]
    if dry_run:
        print("  [DRY RUN] skipping.")
        return None, None, None, None

    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)

    print(f"  Processing: {len(videos)} clips | {n_workers} workers | "
          f"frame_skip={frame_skip} | resize={frame_width}px | complexity={model_complexity}")

    args_list = [(str(v), label_index, key_to_split,
                  frame_skip, frame_width, model_complexity) for v in videos]

    X_rows, y_rows, keys_out, splits_out = [], {e: [] for e in all_labels}, [], []
    t0 = time.time()

    with Pool(processes=n_workers) as pool:
        for i, clip_rows in enumerate(pool.imap_unordered(_process_clip, args_list)):
            if (i + 1) % 20 == 0 or i == 0:
                eta = (time.time() - t0) / (i + 1) * (len(videos) - i - 1)
                print(f"  [{i+1}/{len(videos)}] {len(X_rows):,} samples | "
                      f"ETA {eta/60:.1f} min", flush=True)
            for fvec, key, split, label_map in clip_rows:
                X_rows.append(fvec)
                keys_out.append(key)
                splits_out.append(split)
                for err in all_labels:
                    y_rows[err].append(label_map.get(err, 0))

    print(f"\n  Video done: {len(X_rows):,} samples | {(time.time()-t0)/60:.1f} min")

    # Merge shallow squat (Squat only)
    shallow_dir = cfg.get("shallow_squat_dir")
    if shallow_dir:
        print(f"\n  Loading Shallow_Squat_Error_Dataset ...")
        ss_X, ss_y, ss_splits, ss_keys = load_shallow_squat_dataset(
            ex_dir, shallow_dir,
            max_samples=max_clips,
            n_workers=min(n_workers, 4),
            frame_width=frame_width,
            model_complexity=model_complexity,
        )
        if ss_X:
            n_before = len(X_rows)
            X_rows.extend(ss_X)
            splits_out.extend(ss_splits)
            keys_out.extend(ss_keys)
            if "shallow_squat" not in y_rows:
                y_rows["shallow_squat"] = [0] * n_before
            y_rows["shallow_squat"].extend(ss_y.get("shallow_squat", []))
            for err in list(y_rows.keys()):
                if err == "shallow_squat":
                    continue
                gap = len(X_rows) - len(y_rows[err])
                if gap > 0:
                    y_rows[err].extend([0] * gap)

    print(f"\n  Total: {len(X_rows):,} samples")
    return (np.array(X_rows, dtype=np.float32),
            {e: np.array(v, dtype=int) for e, v in y_rows.items()},
            np.array(splits_out), keys_out)


# ══════════════════════════════════════════════════════════════════════════════
# TRAIN + EVALUATE
# ══════════════════════════════════════════════════════════════════════════════

def train_and_evaluate(clf_name: str, X: np.ndarray, y_dict: dict,
                       splits_arr: np.ndarray, save_dir: Path):

    train_mask = splits_arr == "train"
    test_mask  = (splits_arr == "test") | (splits_arr == "val")

    if train_mask.sum() == 0 or test_mask.sum() == 0:
        print("  [warn] No split info — using random 85/15 split")
        idx = np.arange(len(X))
        tr, te = train_test_split(idx, test_size=0.15, random_state=42)
        train_mask = np.zeros(len(X), dtype=bool); train_mask[tr] = True
        test_mask  = np.zeros(len(X), dtype=bool); test_mask[te]  = True

    X_train = X[train_mask]; X_test = X[test_mask]
    y_train = {e: y[train_mask] for e, y in y_dict.items()}
    y_test  = {e: y[test_mask]  for e, y in y_dict.items()}

    print(f"\n  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    clf = FormClassifier(clf_name)
    clf.train(X_train, y_train)

    print(f"\n  ── Evaluation ──────────────────────────────────────────────")
    f1s = []
    for error_name in y_test:
        if error_name not in clf.pipelines:
            continue
        pipe   = clf.pipelines[error_name]
        X_sc   = pipe["scaler"].transform(X_test)
        y_pred = pipe["svm"].predict(X_sc)
        y_true = y_test[error_name]
        f1     = f1_score(y_true, y_pred, zero_division=0)
        f1s.append(f1)
        print(f"\n  [{error_name}]  F1 = {f1:.3f}")
        print(classification_report(y_true, y_pred,
              target_names=["no error", "error"], zero_division=0, digits=3))
        print(f"  Confusion matrix:\n{confusion_matrix(y_true, y_pred)}")

    if f1s:
        print(f"\n  Mean F1: {np.mean(f1s):.3f}")

    save_dir.mkdir(parents=True, exist_ok=True)
    out = save_dir / f"{clf_name.lower().replace(' ', '_')}_svm.pkl"
    clf.save(str(out))
    print(f"  Saved → {out}")
    return out


# ══════════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

def run_exercise(folder: str, data_dir: Path, save_dir: Path,
                 max_samples: int = None, dry_run: bool = False,
                 n_workers: int = None, frame_skip: int = FRAME_SKIP,
                 frame_width: int = FRAME_WIDTH,
                 model_complexity: int = MODEL_COMPLEXITY):

    cfg    = EXERCISE_CONFIG[folder]
    ex_dir = data_dir / folder

    print(f"\n{'═'*62}")
    print(f"  {cfg['clf_name']}   (folder: {folder})")
    print(f"{'═'*62}")

    if not ex_dir.exists():
        print(f"  [error] Directory not found: {ex_dir}")
        return

    kw = dict(max_samples=max_samples, dry_run=dry_run,
               n_workers=n_workers or 4, frame_width=frame_width,
               model_complexity=model_complexity)

    if cfg["data_mode"] == "image":
        X, y_dict, splits_arr, keys = build_image_dataset(ex_dir, cfg, **kw)
    else:
        X, y_dict, splits_arr, keys = build_video_dataset(
            ex_dir, cfg, max_clips=max_samples, dry_run=dry_run,
            n_workers=n_workers, frame_skip=frame_skip,
            frame_width=frame_width, model_complexity=model_complexity)

    if X is None:
        return

    train_and_evaluate(cfg["clf_name"], X, y_dict, splits_arr, save_dir)


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fitness-AQA Training Pipeline (Optimized)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py --exercise Squat --max_samples 50       # quick sanity check
  python train.py --exercise all                          # full training
  python train.py --exercise Squat --workers 8 --frame_skip 4  # max speed
  python train.py --exercise Squat --frame_skip 1 --complexity 1  # max accuracy
  python train.py --exercise all --dry_run                # verify paths only
        """)
    parser.add_argument("--data_dir",    default="data/")
    parser.add_argument("--save_dir",    default="models/")
    parser.add_argument("--exercise",    default="all",
                        choices=["all", "Squat", "OHP", "BarbellRow"])
    parser.add_argument("--max_samples", default=None, type=int,
                        help="Cap clips/images per exercise")
    parser.add_argument("--dry_run",     action="store_true",
                        help="Verify paths without running pose extraction")
    parser.add_argument("--workers",     default=None, type=int,
                        help=f"Worker count (default: cpu_count-1 = {max(1,cpu_count()-1)})")
    parser.add_argument("--frame_skip",  default=FRAME_SKIP, type=int,
                        help=f"Process every Nth frame (default: {FRAME_SKIP})")
    parser.add_argument("--frame_width", default=FRAME_WIDTH, type=int,
                        help=f"Resize width before MediaPipe (default: {FRAME_WIDTH})")
    parser.add_argument("--complexity",  default=MODEL_COMPLEXITY, type=int,
                        choices=[0, 1, 2],
                        help=f"MediaPipe model complexity (default: {MODEL_COMPLEXITY})")
    args = parser.parse_args()

    exercises = list(EXERCISE_CONFIG) if args.exercise == "all" else [args.exercise]

    print(f"\n  Workers:      {args.workers or max(1, cpu_count()-1)}")
    print(f"  Frame skip:   every {args.frame_skip} frames")
    print(f"  Frame width:  {args.frame_width}px")
    print(f"  Complexity:   {args.complexity}")

    t0 = time.time()
    for ex in exercises:
        run_exercise(
            ex,
            data_dir=Path(args.data_dir),
            save_dir=Path(args.save_dir),
            max_samples=args.max_samples,
            dry_run=args.dry_run,
            n_workers=args.workers,
            frame_skip=args.frame_skip,
            frame_width=args.frame_width,
            model_complexity=args.complexity,
        )

    print(f"\n{'═'*62}")
    print(f"  Total time: {(time.time()-t0)/60:.1f} min")
    print(f"  Models → {Path(args.save_dir).resolve()}")
    print(f"{'═'*62}")