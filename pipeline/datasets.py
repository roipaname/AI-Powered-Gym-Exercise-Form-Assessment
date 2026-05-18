"""

Dataset builders for each exercise type.
"""

import time
import numpy as np
from pathlib import Path
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor

from pipeline.config  import FRAME_SKIP, FRAME_WIDTH, MODEL_COMPLEXITY
from pipeline.helpers import (
    load_temporal_labels, load_binary_labels, load_shallow_squat_labels,
    load_split_keys, build_binary_label_index,
)
from pipeline.workers import (
    _process_clip_temporal, _process_clip_per_rep, _process_image,
)


# ══════════════════════════════════════════════════════════════════════════════
# VIDEO DATASET  (Squat temporal + OHP per-rep)
# ══════════════════════════════════════════════════════════════════════════════

def build_video_dataset(ex_dir: Path, cfg: dict,
                        max_clips: int = None, dry_run: bool = False,
                        n_workers: int = None,
                        frame_skip: int = FRAME_SKIP,
                        frame_width: int = FRAME_WIDTH,
                        model_complexity: int = MODEL_COMPLEXITY):
    video_dir  = ex_dir / cfg["video_dir"]
    labels_dir = ex_dir / cfg["labels_dir"]
    splits_dir = ex_dir / cfg["splits_dir"]
    label_mode = cfg.get("label_mode", "per_rep")

    if not video_dir.exists():
        print(f"  [error] Not found: {video_dir}  ← extract videos.zip here")
        return None, None, None, None

    # ── Labels ───────────────────────────────────────────────────────────────
    print(f"\n  Loading labels  (mode: {label_mode}) ...")
    if label_mode == "temporal":
        all_labels = load_temporal_labels(labels_dir, cfg["label_files"])
    else:
        all_labels = load_binary_labels(labels_dir, cfg["label_files"])

    if not all_labels and not cfg.get("shallow_squat_dir"):
        print("  [error] No labels loaded.")
        return None, None, None, None

    error_names = list(all_labels.keys()) if all_labels else []

    # ── Splits ────────────────────────────────────────────────────────────────
    print(f"\n  Loading splits ...")
    ignore        = cfg.get("splits_ignore", set())
    active_splits = {k: v for k, v in cfg["split_files"].items()
                     if v not in ignore}
    splits        = load_split_keys(splits_dir, active_splits)
    key_to_split  = {}
    for sname, keyset in splits.items():
        for k in keyset:
            key_to_split[k] = sname

    # ── Build label index (per-rep only) ──────────────────────────────────────
    if label_mode == "per_rep" and all_labels:
        label_index = build_binary_label_index(all_labels)
        print(f"  Label index: {len(label_index):,} video IDs")

    # ── Gather videos ─────────────────────────────────────────────────────────
    videos = sorted(video_dir.glob("*.mp4")) + sorted(video_dir.glob("*.avi"))
    print(f"\n  Found {len(videos):,} video files")
    if max_clips:
        videos = videos[:max_clips]
    if dry_run:
        print("  [DRY RUN] skipping extraction.")
        return None, None, None, None

    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)

    print(f"  Workers={n_workers} | frame_skip={frame_skip} | "
          f"resize={frame_width}px | complexity={model_complexity}")

    # ── Build args list ───────────────────────────────────────────────────────
    if label_mode == "temporal":
        args_list = []
        for v in videos:
            clip_key       = v.stem
            clip_intervals = {
                err: all_labels[err].get(clip_key, [])
                for err in error_names
            }
            split_name = key_to_split.get(clip_key, "train")
            args_list.append((
                str(v), clip_intervals, split_name,
                frame_skip, frame_width, model_complexity,
            ))
        worker_fn = _process_clip_temporal
    else:
        args_list = [
            (str(v), label_index, key_to_split,
             frame_skip, frame_width, model_complexity)
            for v in videos
        ]
        worker_fn = _process_clip_per_rep

    # ── Parallel extraction ───────────────────────────────────────────────────
    X_rows     = []
    y_rows     = {e: [] for e in error_names}
    keys_out   = []
    splits_out = []
    t0 = time.time()

    with Pool(processes=n_workers) as pool:
        for i, clip_rows in enumerate(
                pool.imap_unordered(worker_fn, args_list)):
            if (i + 1) % 20 == 0 or i == 0:
                elapsed = time.time() - t0
                eta     = elapsed / (i + 1) * (len(videos) - i - 1)
                print(f"  [{i+1}/{len(videos)}] {len(X_rows):,} samples | "
                      f"ETA {eta/60:.1f} min", flush=True)
            for fvec, key, split, label_map in clip_rows:
                X_rows.append(fvec)
                keys_out.append(key)
                splits_out.append(split)
                for err in error_names:
                    y_rows[err].append(label_map.get(err, 0))

    print(f"\n  Video done: {len(X_rows):,} samples | "
          f"{(time.time()-t0)/60:.1f} min")

    # ── Merge ShallowSquat (Squat only) ──────────────────────────────────────
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
    if not X_rows:
        return None, None, None, None

    return (np.array(X_rows, dtype=np.float32),
            {e: np.array(v, dtype=int) for e, v in y_rows.items()},
            np.array(splits_out),
            keys_out)


# ══════════════════════════════════════════════════════════════════════════════
# IMAGE DATASET  (BarbellRow)
# ══════════════════════════════════════════════════════════════════════════════

def build_image_dataset(ex_dir: Path, cfg: dict,
                        max_samples: int = None, dry_run: bool = False,
                        n_workers: int = 4,
                        frame_width: int = FRAME_WIDTH,
                        model_complexity: int = MODEL_COMPLEXITY):
    image_dir  = ex_dir / cfg["image_dir"]
    labels_dir = ex_dir / cfg["labels_dir"]

    if not image_dir.exists():
        print(f"  [error] Not found: {image_dir}  ← extract images zip here")
        return None, None, None, None

    print(f"\n  Loading labels ...")
    all_labels = load_binary_labels(labels_dir, cfg["label_files"])
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

    args_list = [
        (str(p), all_labels, key_to_split, frame_width, model_complexity)
        for p in images
    ]

    X_rows     = []
    y_rows     = {e: [] for e in all_labels}
    keys_out   = []
    splits_out = []
    no_pose    = 0
    t0 = time.time()

    print(f"  Processing on {n_workers} threads ...")
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        for i, result in enumerate(pool.map(_process_image, args_list)):
            if (i + 1) % 500 == 0:
                eta = (time.time() - t0) / (i + 1) * (len(images) - i - 1)
                print(f"  [{i+1}/{len(images)}] ETA {eta/60:.1f} min",
                      flush=True)
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

    if not X_rows:
        return None, None, None, None

    return (np.array(X_rows, dtype=np.float32),
            {e: np.array(v, dtype=int) for e, v in y_rows.items()},
            np.array(splits_out),
            keys_out)


# ══════════════════════════════════════════════════════════════════════════════
# SHALLOW SQUAT DATASET
# ══════════════════════════════════════════════════════════════════════════════

def load_shallow_squat_dataset(ex_dir: Path, shallow_dir_rel: str,
                               max_samples: int = None, dry_run: bool = False,
                               n_workers: int = 4,
                               frame_width: int = FRAME_WIDTH,
                               model_complexity: int = MODEL_COMPLEXITY):
    """
    Loads the Shallow_Squat_Error_Dataset — an image-based mini dataset
    inside the Squat folder.

    Layout:
        images/                     ← VIDEOID_PERSONID_REPIDX.jpg
        splits/
            train_ids.json / val_ids.json / test_ids.json
        labels_shallow_depth.json   ← {key: 0/1}
    """
    s_dir = ex_dir / shallow_dir_rel
    if not s_dir.exists():
        print(f"    [warn] Not found: {s_dir} — skipping shallow squat.")
        return [], {}, [], []

    label_dict = load_shallow_squat_labels(s_dir)
    if not label_dict:
        print(f"    [warn] No label data in {s_dir}")
        return [], {}, [], []

    # Load splits
    key_to_split = {}
    splits_dir   = s_dir / "splits"
    if splits_dir.exists():
        for sname, fname in [("train", "train_ids.json"),
                              ("val",   "val_ids.json"),
                              ("test",  "test_ids.json")]:
            from pipeline.helpers import load_json
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
    args_list   = [
        (str(p), label_dicts, key_to_split, frame_width, model_complexity)
        for p in images
    ]

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
