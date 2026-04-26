"""
train.py — Fitness-AQA Full Training Pipeline
==============================================

Handles the EXACT dataset layout:

data/
├── Squat/
│   ├── Labeled_Dataset/
│   │   ├── splits/
│   │   │   ├── train_keys.json       ← list of "VIDEOID_PERSONID_REPIDX" strings
│   │   │   ├── val_keys.json
│   │   │   └── test_keys.json
│   │   └── labels/
│   │       ├── error_knees_inward.json   ← {"VIDEOID_PERSONID_REPIDX": 0/1, ...}
│   │       ├── error_knees_forward.json
│   │       ├── error_rounded_back.json
│   │       └── error_shallow_squat.json
│   └── videos.zip  → extract to Squat/videos/
│
├── OHP/
│   ├── Labeled_Dataset/
│   │   ├── splits/
│   │   │   ├── train_keys.json
│   │   │   ├── val_keys.json
│   │   │   └── test_keys.json
│   │   └── labels/
│   │       ├── error_elbows.json
│   │       └── error_knees.json
│   └── videos.zip  → extract to OHP/videos/
│
└── BarbellRow/
    ├── Labeled_Dataset/
    │   ├── Splits/
    │   │   ├── Splits_Lumbar_error/
    │   │   │   ├── train_Ids.json
    │   │   │   ├── val_Ids.json
    │   │   │   └── test_Ids.json
    │   │   └── Splits_TorsoAngle_error/
    │   │       ├── train_Ids.json
    │   │       ├── val_Ids.json
    │   │       └── test_Ids.json
    │   └── labels/
    │       ├── labels_lumbar_error.json
    │       └── labels_torso_angle_error.json
    └── BarbellRow_images.zip  → extract to BarbellRow/images/
        (images named VIDEOID_PERSONID_REPIDX.jpg, e.g. 62724_7_9.jpg)

Key insight for BarbellRow:
  - Data is IMAGES not videos. Each image = one rep frame.
  - Filename IS the label key: "62724_7_9.jpg" -> key "62724_7_9"
  - Feature extraction uses single-frame pose (no segmentation needed)

Usage:
  python train.py --exercise all
  python train.py --exercise BarbellRow --max_samples 300
  python train.py --exercise Squat --max_samples 100
  python train.py --exercise all --dry_run
"""

import argparse
import json
import cv2
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import time

from modules.extractor  import PoseExtractor
from modules.segmentor  import RepSegmentor
from modules.features   import compute_rep_features, feature_vector, FEATURE_KEYS
from modules.classifier import FormClassifier


# ══════════════════════════════════════════════════════════════════════════════
# EXERCISE CONFIG — update label filenames here if yours differ
# ══════════════════════════════════════════════════════════════════════════════

EXERCISE_CONFIG = {

    "Squat": {
        "clf_name":   "BackSquat",
        "data_mode":  "video",
        "video_dir":  "videos",
        "splits_dir": "Labeled_Dataset/splits",
        "split_files": {
            "train": "train_keys.json",
            "val":   "val_keys.json",
            "test":  "test_keys.json",
        },
        # traj_nan.json is a trajectory artifact — NOT a split file, skip it
        "splits_ignore": {"traj_nan.json"},
        "labels_dir":  "Labeled_Dataset/labels",
        "label_files": {
            # Flat label JSONs in Labeled_Dataset/labels/
            "knees_forward": "error_knees_forward.json",
            "knees_inward":  "error_knees_inward.json",
            # shallow_squat lives in its own subfolder — loaded separately below
        },
        # Shallow_Squat_Error_Dataset subfolder: may have its own splits + label file.
        # Set path relative to ex_dir, or None to skip.
        "shallow_squat_dir": "Labeled_Dataset/labels/Shallow_Squat_Error_Dataset",
        "per_error_splits": None,
    },

    "OHP": {
        "clf_name":   "OverheadPress",
        "data_mode":  "video",
        "video_dir":  "videos",
        "splits_dir": "Labeled_Dataset/splits",
        "split_files": {
            "train": "train_keys.json",
            "val":   "val_keys.json",
            "test":  "test_keys.json",
        },
        "splits_ignore": set(),
        "labels_dir":  "Labeled_Dataset/labels",
        "label_files": {
            "elbow_error": "error_elbows.json",
            "knees_error": "error_knees.json",
        },
        "shallow_squat_dir": None,
        "per_error_splits": None,
    },

    "BarbellRow": {
        "clf_name":   "BarbellRow",
        "data_mode":  "image",
        "image_dir":  "images",
        "splits_dir": None,
        "split_files": None,
        "labels_dir": "Labeled_Dataset/labels",
        "label_files": {
            "lumbar_error":      "labels_lumbar_error.json",
            "torso_angle_error": "labels_torso_angle_error.json",
        },
        # BarbellRow has one split folder per error
        "per_error_splits": {
            "lumbar_error": {
                "splits_dir":  "Labeled_Dataset/Splits/Splits_Lumbar_error",
                "split_files": {
                    "train": "train_Ids.json",
                    "val":   "val_Ids.json",
                    "test":  "test_Ids.json",
                },
            },
            "torso_angle_error": {
                "splits_dir":  "Labeled_Dataset/Splits/Splits_TorsoAngle_error",
                "split_files": {
                    "train": "train_Ids.json",
                    "val":   "val_Ids.json",
                    "test":  "test_Ids.json",
                },
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


def load_labels(labels_dir: Path, label_files: dict) -> dict:
    """
    Returns labels[error_name][clip_key] = 0 or 1
    clip_key = "VIDEOID_PERSONID_REPIDX"
    """
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
            if isinstance(v, (int, float)):
                # Normal: {"key": 0} or {"key": 1}
                parsed[str(k)] = int(v)
            elif isinstance(v, list):
                # List value: [1] or [0, ...] — take first element
                if len(v) > 0 and isinstance(v[0], (int, float)):
                    parsed[str(k)] = int(v[0])
                else:
                    skipped += 1
            elif isinstance(v, dict):
                # Nested dict: {"label": 1, ...} — grab first numeric value
                inner = [x for x in v.values() if isinstance(x, (int, float))]
                if inner:
                    parsed[str(k)] = int(inner[0])
                else:
                    skipped += 1
            else:
                skipped += 1

        if skipped:
            print(f"    [warn] {error_name}: skipped {skipped} unreadable entries")
        if not parsed:
            print(f"    [warn] {error_name}: no valid entries parsed — skipping")
            continue

        labels[error_name] = parsed
        pos   = sum(parsed.values())
        total = len(parsed)
        print(f"    {error_name}: {total:,} entries | {pos} positive ({100*pos/total:.1f}%)")
    return labels


def load_split_keys(splits_dir: Path, split_files: dict) -> dict:
    """Returns splits["train"/"val"/"test"] = set of key strings."""
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


# ══════════════════════════════════════════════════════════════════════════════
# SINGLE-FRAME FEATURE EXTRACTION  (BarbellRow — image mode)
# ══════════════════════════════════════════════════════════════════════════════

def angle_3pt(a, b, c) -> float:
    ba = a - b; bc = c - b
    n1 = np.linalg.norm(ba); n2 = np.linalg.norm(bc)
    if n1 < 1e-8 or n2 < 1e-8:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(ba, bc) / (n1 * n2), -1, 1))))


def single_frame_features(image_bgr: np.ndarray, extractor: PoseExtractor):
    """
    Pose features from one image frame.
    Temporal features are zeroed (not available from a single frame).
    Returns feature dict or None if pose not detected.
    """
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    lm, _ = extractor.extract(rgb)
    if lm is None:
        return None

    xyz = lm[:, :3]   # drop visibility

    knee_L  = angle_3pt(xyz[23], xyz[25], xyz[27])
    knee_R  = angle_3pt(xyz[24], xyz[26], xyz[28])
    hip_L   = angle_3pt(xyz[11], xyz[23], xyz[25])
    hip_R   = angle_3pt(xyz[12], xyz[24], xyz[26])
    elbow_L = angle_3pt(xyz[11], xyz[13], xyz[15])
    elbow_R = angle_3pt(xyz[12], xyz[14], xyz[16])
    spine   = angle_3pt(xyz[11], xyz[23], xyz[25])

    return {
        "knee_angle_min":    min(knee_L, knee_R),
        "knee_angle_max":    max(knee_L, knee_R),
        "knee_angle_range":  abs(knee_L - knee_R),
        "hip_angle_min":     min(hip_L, hip_R),
        "hip_angle_max":     max(hip_L, hip_R),
        "elbow_angle_min":   min(elbow_L, elbow_R),
        "elbow_angle_max":   max(elbow_L, elbow_R),
        "spine_angle_min":   spine,
        "spine_angle_mean":  spine,
        "knee_symmetry":     abs(knee_L - knee_R),
        "hip_symmetry":      abs(hip_L - hip_R),
        "elbow_symmetry":    abs(elbow_L - elbow_R),
        "depth_range":       float(abs(lm[23, 1] - lm[24, 1])),
        "knee_temporal_var": 0.0,   # N/A for single frame
        "hip_temporal_var":  0.0,
        "knee_avg_velocity": 0.0,
    }


# ══════════════════════════════════════════════════════════════════════════════
# IMAGE DATASET BUILDER  (BarbellRow)
# ══════════════════════════════════════════════════════════════════════════════

def build_image_dataset(ex_dir: Path, cfg: dict, max_samples: int = None, dry_run: bool = False):
    image_dir  = ex_dir / cfg["image_dir"]
    labels_dir = ex_dir / cfg["labels_dir"]

    if not image_dir.exists():
        print(f"  [error] Image folder not found: {image_dir}")
        print(f"          Unzip BarbellRow_images.zip into that folder.")
        return None, None, None, None

    print(f"\n  Loading labels ...")
    all_labels = load_labels(labels_dir, cfg["label_files"])
    if not all_labels:
        return None, None, None, None

    # Load per-error splits and build unified key→split map
    print(f"\n  Loading splits ...")
    key_to_split = {}
    for error_name, split_cfg in cfg["per_error_splits"].items():
        s_dir   = ex_dir / split_cfg["splits_dir"]
        splits  = load_split_keys(s_dir, split_cfg["split_files"])
        for sname, keyset in splits.items():
            for k in keyset:
                # First-seen split wins (lumbar loaded first → primary)
                if k not in key_to_split:
                    key_to_split[k] = sname

    # All images
    images = (sorted(image_dir.glob("*.jpg")) +
              sorted(image_dir.glob("*.jpeg")) +
              sorted(image_dir.glob("*.png")))
    print(f"\n  Found {len(images):,} images")

    if max_samples:
        images = images[:max_samples]
        print(f"  Capped at {max_samples}")

    if dry_run:
        print(f"  [DRY RUN] skipping pose extraction.")
        return None, None, None, None

    extractor = PoseExtractor()
    X_rows, y_rows, keys_out, splits_out = [], {e: [] for e in all_labels}, [], []
    no_label = no_pose = 0

    for i, img_path in enumerate(images):
        key = img_path.stem   # "62724_7_9"

        if not any(key in lbl for lbl in all_labels.values()):
            no_label += 1
            continue

        if (i + 1) % 500 == 0:
            print(f"  [{i+1}/{len(images)}]", flush=True)

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        feats = single_frame_features(img, extractor)
        if feats is None:
            no_pose += 1
            continue

        X_rows.append(feature_vector(feats))
        keys_out.append(key)
        splits_out.append(key_to_split.get(key, "train"))

        for error_name, lbl_dict in all_labels.items():
            y_rows[error_name].append(lbl_dict.get(key, 0))

    extractor.close()

    print(f"\n  Collected:      {len(X_rows):,} samples")
    print(f"  Skipped (no label): {no_label:,}")
    print(f"  Skipped (no pose):  {no_pose:,}")

    X          = np.array(X_rows, dtype=np.float32)
    y_dict     = {e: np.array(v, dtype=int) for e, v in y_rows.items()}
    splits_arr = np.array(splits_out)
    return X, y_dict, splits_arr, keys_out


# ══════════════════════════════════════════════════════════════════════════════
# VIDEO DATASET BUILDER  (Squat / OHP)
# ══════════════════════════════════════════════════════════════════════════════

def load_shallow_squat_dataset(ex_dir: Path, shallow_dir_rel: str,
                               extractor: "PoseExtractor",
                               max_samples: int = None, dry_run: bool = False):
    """
    Shallow_Squat_Error_Dataset is a self-contained image dataset, structured
    identically to BarbellRow:

      Shallow_Squat_Error_Dataset/
        images/           <- VIDEOID_PERSONID_REPIDX.jpg  (e.g. 32903_8_40.jpg)
        splits/
          train_ids.json  <- list of "VIDEOID_PERSONID_REPIDX" keys  (lowercase!)
          val_ids.json
          test_ids.json
        labels_shallow_depth.json   <- {key: 0/1}

    Returns (X_rows, y_rows, splits_out, keys_out) — lists ready to append
    to the main Squat dataset, or ([], {}, [], []) on failure/skip.
    """
    s_dir = ex_dir / shallow_dir_rel
    if not s_dir.exists():
        print(f"    [warn] Shallow_Squat_Error_Dataset not found: {s_dir} — skipping.")
        return [], {}, [], []

    # ── Find the label JSON ───────────────────────────────────────────────────
    # Filename is labels_shallow_depth.json (confirmed from uploaded file)
    label_candidates = list(s_dir.glob("labels_*.json")) + list(s_dir.glob("*.json"))
    label_candidates = [f for f in label_candidates
                        if "readme" not in f.name.lower()
                        and f.parent == s_dir]   # only top-level files
    if not label_candidates:
        print(f"    [warn] No label JSON found in {s_dir}")
        return [], {}, [], []

    label_dict = {}
    for fpath in label_candidates:
        data = load_json(fpath)
        if not isinstance(data, dict) or not data:
            continue
        for k, v in data.items():
            if isinstance(v, (int, float)):
                label_dict[str(k)] = int(v)
            elif isinstance(v, list) and v and isinstance(v[0], (int, float)):
                label_dict[str(k)] = int(v[0])
            elif isinstance(v, dict):
                inner = [x for x in v.values() if isinstance(x, (int, float))]
                if inner:
                    label_dict[str(k)] = int(inner[0])
        pos = sum(label_dict.values())
        total = len(label_dict)
        print(f"    shallow_squat label ({fpath.name}): "
              f"{total:,} entries, {pos} positive ({100*pos/total:.1f}%)" if total else "")

    if not label_dict:
        print(f"    [warn] shallow_squat label JSON was empty.")
        return [], {}, [], []

    # ── Load splits (train_ids.json / val_ids.json / test_ids.json) ──────────
    splits_dir = s_dir / "splits"
    key_to_split = {}
    if splits_dir.exists():
        # Filenames use lowercase _ids: train_ids.json, val_ids.json, test_ids.json
        for split_name, fname in [("train", "train_ids.json"),
                                   ("val",   "val_ids.json"),
                                   ("test",  "test_ids.json")]:
            data = load_json(splits_dir / fname)
            if data:
                for k in data:
                    key_to_split[str(k)] = split_name
                print(f"    shallow_squat split '{split_name}': {len(data):,} keys")
    else:
        print(f"    [warn] No splits/ folder in shallow squat dir — all assigned to train.")

    # ── Find images ───────────────────────────────────────────────────────────
    img_dir = s_dir / "images"
    if not img_dir.exists():
        print(f"    [warn] No images/ folder found in {s_dir}")
        print(f"           Extract the shallow squat images zip into {img_dir}")
        return [], {}, [], []

    images = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.jpeg")) + sorted(img_dir.glob("*.png"))
    print(f"    shallow_squat images: {len(images):,} found")

    if max_samples:
        images = images[:max_samples]

    if dry_run:
        print(f"    [DRY RUN] skipping shallow squat image extraction.")
        return [], {}, [], []

    # ── Extract features ──────────────────────────────────────────────────────
    X_rows, splits_out, keys_out = [], [], []
    y_rows = {"shallow_squat": []}
    no_label = no_pose = 0

    for i, img_path in enumerate(images):
        key = img_path.stem   # "32903_8_40"

        if key not in label_dict:
            no_label += 1
            continue

        if (i + 1) % 500 == 0:
            print(f"    shallow_squat [{i+1}/{len(images)}]", flush=True)

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        feats = single_frame_features(img, extractor)
        if feats is None:
            no_pose += 1
            continue

        X_rows.append(feature_vector(feats))
        keys_out.append(key)
        splits_out.append(key_to_split.get(key, "train"))
        y_rows["shallow_squat"].append(label_dict[key])

    print(f"    shallow_squat collected: {len(X_rows):,} | "
          f"no_label: {no_label} | no_pose: {no_pose}")

    return X_rows, y_rows, splits_out, keys_out


def build_video_dataset(ex_dir: Path, cfg: dict, max_clips: int = None, dry_run: bool = False):
    video_dir  = ex_dir / cfg["video_dir"]
    labels_dir = ex_dir / cfg["labels_dir"]
    splits_dir = ex_dir / cfg["splits_dir"]

    if not video_dir.exists():
        print(f"  [error] Video folder not found: {video_dir}")
        print(f"          Unzip videos.zip into that folder.")
        return None, None, None, None

    print(f"\n  Loading labels ...")
    all_labels = load_labels(labels_dir, cfg["label_files"])

    if not all_labels and not cfg.get("shallow_squat_dir"):
        print("  [error] No labels loaded at all.")
        return None, None, None, None

    print(f"\n  Loading splits ...")
    # Exclude non-split artifact files like traj_nan.json
    ignore = cfg.get("splits_ignore", set())
    active_splits = {k: v for k, v in cfg["split_files"].items() if v not in ignore}
    splits = load_split_keys(splits_dir, active_splits)
    key_to_split = {}
    for sname, keyset in splits.items():
        for k in keyset:
            key_to_split[k] = sname

    videos = sorted(video_dir.glob("*.mp4")) + sorted(video_dir.glob("*.avi"))
    print(f"\n  Found {len(videos):,} video files")

    if max_clips:
        videos = videos[:max_clips]

    if dry_run:
        print(f"  [DRY RUN] skipping pose extraction.")
        return None, None, None, None

    extractor = PoseExtractor()
    X_rows, y_rows, keys_out, splits_out = [], {e: [] for e in all_labels}, [], []
    t_total = 0

    for vi, vpath in enumerate(videos):
        base_id = vpath.stem.split("_")[0]   # "55323" from "55323_15.mp4"
        t0  = time.time()
        cap = cv2.VideoCapture(str(vpath))
        if not cap.isOpened():
            continue

        segmentor = RepSegmentor()
        lm_buf    = []
        while True:
            ret, frame = cap.read()
            if not ret: break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            lm, _ = extractor.extract(rgb)
            if lm is not None:
                segmentor.update(lm)
                lm_buf.append(lm)
        cap.release()

        reps = segmentor.get_reps()
        t_total += time.time() - t0
        matched = 0

        for rep_obj in reps:
            rep_idx = rep_obj.rep_number - 1   # 0-indexed

            # Find label key: VIDEOID_PERSONID_REPIDX
            found_key = None
            for lbl_dict in all_labels.values():
                for k in lbl_dict:
                    parts = k.split("_")
                    if len(parts) == 3 and parts[0] == base_id and int(parts[2]) == rep_idx:
                        found_key = k
                        break
                if found_key:
                    break

            if not found_key:
                continue

            seq = lm_buf[rep_obj.start_frame: rep_obj.end_frame + 1]
            if len(seq) < 5:
                continue

            try:
                feats = compute_rep_features(seq)
            except Exception:
                continue

            X_rows.append(feature_vector(feats))
            keys_out.append(found_key)
            splits_out.append(key_to_split.get(found_key, "train"))
            for error_name, lbl_dict in all_labels.items():
                y_rows[error_name].append(lbl_dict.get(found_key, 0))
            matched += 1

        if (vi + 1) % 50 == 0 or vi == 0:
            eta = (t_total / (vi + 1)) * (len(videos) - vi - 1)
            print(f"  [{vi+1}/{len(videos)}] {vpath.name} | "
                  f"{len(reps)} reps, {matched} matched | ETA {eta/60:.1f} min")

    # ── Merge Shallow Squat image dataset (Squat only) ──────────────────────
    shallow_dir = cfg.get("shallow_squat_dir")
    if shallow_dir:
        print(f"\n  Loading Shallow_Squat_Error_Dataset (image-based) ...")
        ss_X, ss_y, ss_splits, ss_keys = load_shallow_squat_dataset(
            ex_dir, shallow_dir, extractor,
            max_samples=max_clips,   # reuse same cap for consistency
            dry_run=False,
        )
        if ss_X:
            X_rows.extend(ss_X)
            splits_out.extend(ss_splits)
            keys_out.extend(ss_keys)
            # Merge y_rows: shallow_squat is a new error column;
            # pad existing video rows with 0 (unknown/no shallow error)
            n_video = len(X_rows) - len(ss_X)
            if "shallow_squat" not in y_rows:
                y_rows["shallow_squat"] = [0] * n_video
            y_rows["shallow_squat"].extend(ss_y.get("shallow_squat", []))
            # For all OTHER error columns, pad the new shallow squat rows with 0
            for err in list(y_rows.keys()):
                if err == "shallow_squat":
                    continue
                current_len = len(y_rows[err])
                if current_len < len(X_rows):
                    y_rows[err].extend([0] * (len(X_rows) - current_len))

    extractor.close()
    print(f"\n  Total collected: {len(X_rows):,} samples "
          f"({len(videos)} video clips + shallow squat images)")

    X          = np.array(X_rows, dtype=np.float32)
    y_dict     = {e: np.array(v, dtype=int) for e, v in y_rows.items()}
    splits_arr = np.array(splits_out)
    return X, y_dict, splits_arr, keys_out


# ══════════════════════════════════════════════════════════════════════════════
# TRAIN + EVALUATE
# ══════════════════════════════════════════════════════════════════════════════

def train_and_evaluate(clf_name: str, X: np.ndarray, y_dict: dict,
                       splits_arr: np.ndarray, save_dir: Path):

    train_mask = splits_arr == "train"
    test_mask  = (splits_arr == "test") | (splits_arr == "val")

    # Fallback to random split if split info is absent
    if train_mask.sum() == 0 or test_mask.sum() == 0:
        print("  [warn] No split info found — using random 85/15 split")
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
        cm = confusion_matrix(y_true, y_pred)
        print(f"\n  [{error_name}]  F1 = {f1:.3f}")
        print(classification_report(y_true, y_pred,
              target_names=["no error", "error"], zero_division=0, digits=3))
        print(f"  Confusion matrix:\n{cm}")

    if f1s:
        print(f"\n  Mean F1: {np.mean(f1s):.3f}")

    save_dir.mkdir(parents=True, exist_ok=True)
    out = save_dir / f"{clf_name.lower().replace(' ','_')}_svm.pkl"
    clf.save(str(out))
    print(f"  Saved → {out}")
    return out


# ══════════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

def run_exercise(folder: str, data_dir: Path, save_dir: Path,
                 max_samples: int = None, dry_run: bool = False):
    cfg    = EXERCISE_CONFIG[folder]
    ex_dir = data_dir / folder

    print(f"\n{'═'*62}")
    print(f"  {cfg['clf_name']}   (folder: {folder})")
    print(f"{'═'*62}")

    if not ex_dir.exists():
        print(f"  [error] Directory not found: {ex_dir}")
        return

    if cfg["data_mode"] == "image":
        X, y_dict, splits_arr, keys = build_image_dataset(
            ex_dir, cfg, max_samples=max_samples, dry_run=dry_run)
    else:
        X, y_dict, splits_arr, keys = build_video_dataset(
            ex_dir, cfg, max_clips=max_samples, dry_run=dry_run)

    if X is None:
        return

    train_and_evaluate(cfg["clf_name"], X, y_dict, splits_arr, save_dir)


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fitness-AQA Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py --exercise all
  python train.py --exercise BarbellRow --max_samples 300
  python train.py --exercise Squat --max_samples 100
  python train.py --exercise all --dry_run
        """)
    parser.add_argument("--data_dir",    default="data/")
    parser.add_argument("--save_dir",    default="models/")
    parser.add_argument("--exercise",    default="all",
                        choices=["all", "Squat", "OHP", "BarbellRow"])
    parser.add_argument("--max_samples", default=None, type=int,
                        help="Cap clips or images per exercise")
    parser.add_argument("--dry_run",     action="store_true",
                        help="Verify paths without running pose extraction")
    args = parser.parse_args()

    exercises = list(EXERCISE_CONFIG) if args.exercise == "all" else [args.exercise]
    t0 = time.time()
    for ex in exercises:
        run_exercise(ex, Path(args.data_dir), Path(args.save_dir),
                     max_samples=args.max_samples, dry_run=args.dry_run)

    print(f"\n  Total time: {(time.time()-t0)/60:.1f} min")