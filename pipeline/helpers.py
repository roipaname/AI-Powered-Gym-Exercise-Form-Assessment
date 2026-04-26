"""
pipeline/helpers.py
Shared utilities: logging suppression, label loaders, index builders,
split loaders, single-frame feature extraction.
"""

import os
import json
import numpy as np
from pathlib import Path


# ── Silence MediaPipe / TF before anything imports them ───────────────────────
def suppress_mediapipe():
    """Call at the top of every worker process and in main."""
    os.environ["GLOG_minloglevel"]       = "3"
    os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
    os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
    import warnings
    warnings.filterwarnings("ignore")


suppress_mediapipe()   # apply immediately on import


# ══════════════════════════════════════════════════════════════════════════════
# JSON / FILE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def load_json(path: Path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def parse_binary_value(v):
    """Convert any label value format (int/float/list/dict) to 0/1 or None."""
    if isinstance(v, (int, float)):
        return int(v)
    if isinstance(v, list) and v and isinstance(v[0], (int, float)):
        return int(v[0])
    if isinstance(v, dict):
        nums = [x for x in v.values() if isinstance(x, (int, float))]
        if nums:
            return int(nums[0])
    return None


# ══════════════════════════════════════════════════════════════════════════════
# LABEL LOADERS
# ══════════════════════════════════════════════════════════════════════════════

def load_temporal_labels(labels_dir: Path, label_files: dict) -> dict:
    """
    Load Squat-style TEMPORAL labels.

    File format:
        key   = "VIDEOID_PERSONID"  (2 parts, matches video stem)
        value = [[start_sec, end_sec], ...]  or  []  (clean clip)

    Returns:
        temporal[error_name][clip_key] = [(start, end), ...]
    """
    temporal = {}
    for error_name, fname in label_files.items():
        fpath = labels_dir / fname
        data  = load_json(fpath)
        if data is None:
            print(f"    [warn] Missing: {fpath}")
            continue

        parsed      = {}
        error_clips = 0
        for k, v in data.items():
            if not isinstance(v, list):
                continue
            intervals = []
            for item in v:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    intervals.append((float(item[0]), float(item[1])))
            parsed[str(k)] = intervals
            if intervals:
                error_clips += 1

        if not parsed:
            print(f"    [warn] {error_name}: no entries parsed from {fname}")
            continue

        temporal[error_name] = parsed
        clean = len(parsed) - error_clips
        print(f"    {error_name}: {len(parsed):,} clips | "
              f"{error_clips} with errors | {clean} clean")

    return temporal


def load_binary_labels(labels_dir: Path, label_files: dict) -> dict:
    """
    Load standard per-rep binary labels.

    File format:
        key   = "VIDEOID_PERSONID_REPIDX"
        value = 0 or 1 (or list/dict variant)

    Returns:
        labels[error_name][clip_key] = 0 or 1
    """
    labels = {}
    for error_name, fname in label_files.items():
        fpath = labels_dir / fname
        data  = load_json(fpath)
        if data is None:
            print(f"    [warn] Missing: {fpath}")
            continue
        parsed  = {}
        skipped = 0
        for k, v in data.items():
            val = parse_binary_value(v)
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
        print(f"    {error_name}: {total:,} entries | "
              f"{pos} positive ({100*pos/total:.1f}%)")
    return labels


def load_shallow_squat_labels(s_dir: Path) -> dict:
    """
    Load shallow squat binary labels from the subfolder's JSON file(s).
    Returns flat {key: 0/1} dict.
    """
    label_dict = {}
    for fpath in [f for f in s_dir.glob("*.json")
                  if "readme" not in f.name.lower()]:
        data = load_json(fpath)
        if isinstance(data, dict):
            for k, v in data.items():
                val = parse_binary_value(v)
                if val is not None:
                    label_dict[str(k)] = val
            if label_dict:
                pos = sum(label_dict.values())
                print(f"    shallow_squat ({fpath.name}): "
                      f"{len(label_dict):,} | {pos} positive")
    return label_dict


# ══════════════════════════════════════════════════════════════════════════════
# SPLIT LOADERS
# ══════════════════════════════════════════════════════════════════════════════

def load_split_keys(splits_dir: Path, split_files: dict) -> dict:
    """Returns {split_name: set_of_key_strings}."""
    splits = {}
    for split_name, fname in split_files.items():
        fpath = splits_dir / fname
        data  = load_json(fpath)
        if data is None:
            print(f"    [warn] Missing split file: {fpath}")
            splits[split_name] = set()
        else:
            splits[split_name] = set(str(k) for k in data)
            print(f"    split '{split_name}': {len(splits[split_name]):,} keys")
    return splits


def build_binary_label_index(all_labels: dict) -> dict:
    """
    Build O(1) lookup for per-rep labels.

    Input:  all_labels[error_name]["VIDEOID_PERSONID_REPIDX"] = 0/1
    Output: index[video_id][rep_idx] = {"_key": str, error1: 0/1, ...}
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
# TEMPORAL HELPER
# ══════════════════════════════════════════════════════════════════════════════

def timestamp_in_intervals(t: float, intervals: list) -> bool:
    """True if timestamp t falls inside any (start, end) interval."""
    return any(s <= t <= e for s, e in intervals)


# ══════════════════════════════════════════════════════════════════════════════
# SINGLE-FRAME FEATURES  (for image datasets)
# ══════════════════════════════════════════════════════════════════════════════

def _angle(a, b, c) -> float:
    ba = a - b; bc = c - b
    n1 = np.linalg.norm(ba); n2 = np.linalg.norm(bc)
    if n1 < 1e-8 or n2 < 1e-8:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(ba, bc) / (n1 * n2), -1, 1))))


def single_frame_features(lm: np.ndarray) -> dict:
    """Extract the full 16-feature dict from a single (33,4) landmark array."""
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
        "knee_temporal_var": 0.0,
        "hip_temporal_var":  0.0,
        "knee_avg_velocity": 0.0,
    }
