"""
train.py — Fitness-AQA Training Entry Point
============================================
All logic is in pipeline/:
  config.py   — exercise config and speed defaults
  helpers.py  — label loaders, split loaders, feature helpers
  workers.py  — multiprocessing worker functions
  datasets.py — dataset builders (video + image)

Usage:
  python train.py --exercise Squat --max_samples 50
  python train.py --exercise all
  python train.py --exercise all --dry_run
  python train.py --exercise Squat --workers 6 --frame_skip 2
"""

# Suppress MediaPipe / TF logs before any imports
from pipeline.helpers import suppress_mediapipe
suppress_mediapipe()

import argparse
import numpy as np
from pathlib import Path
from multiprocessing import cpu_count
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import time

from pipeline.config   import EXERCISE_CONFIG, FRAME_SKIP, FRAME_WIDTH, MODEL_COMPLEXITY
from pipeline.datasets import build_video_dataset, build_image_dataset
from modules.classifier import FormClassifier


# ══════════════════════════════════════════════════════════════════════════════
# TRAIN + EVALUATE
# ══════════════════════════════════════════════════════════════════════════════

def train_and_evaluate(clf_name: str, X: np.ndarray, y_dict: dict,
                       splits_arr: np.ndarray, save_dir: Path):

    train_mask = splits_arr == "train"
    test_mask  = (splits_arr == "test") | (splits_arr == "val")

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
    from modules.classifier import DETECTION_THRESHOLD
    for error_name in y_test:
        if error_name not in clf.pipelines:
            continue
        pipe   = clf.pipelines[error_name]
        X_sc   = pipe["scaler"].transform(X_test)
        # Use same threshold as predict() — lower than 0.5 for imbalanced data
        probs  = pipe["svm"].predict_proba(X_sc)[:, 1]
        y_pred = (probs >= DETECTION_THRESHOLD).astype(int)
        y_true = y_test[error_name]
        pos_train = int(y_train[error_name].sum())
        pos_test  = int(y_true.sum())
        f1     = f1_score(y_true, y_pred, zero_division=0)
        f1s.append(f1)
        print(f"\n  [{error_name}]  F1 = {f1:.3f}  "
              f"(train pos: {pos_train} | test pos: {pos_test} | threshold: {DETECTION_THRESHOLD})")
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

    if cfg["data_mode"] == "image":
        X, y_dict, splits_arr, keys = build_image_dataset(
            ex_dir, cfg,
            max_samples=max_samples, dry_run=dry_run,
            n_workers=n_workers or 4,
            frame_width=frame_width,
            model_complexity=model_complexity,
        )
    else:
        X, y_dict, splits_arr, keys = build_video_dataset(
            ex_dir, cfg,
            max_clips=max_samples, dry_run=dry_run,
            n_workers=n_workers,
            frame_skip=frame_skip,
            frame_width=frame_width,
            model_complexity=model_complexity,
        )

    if X is None:
        print("  [skip] No samples collected — skipping training.")
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
  python train.py --exercise Squat --max_samples 50    # quick sanity check
  python train.py --exercise all                        # full training
  python train.py --exercise Squat --workers 6 --frame_skip 2
  python train.py --exercise all --dry_run
        """)
    parser.add_argument("--data_dir",    default="data/")
    parser.add_argument("--save_dir",    default="models/")
    parser.add_argument("--exercise",    default="all",
                        choices=["all", "Squat", "OHP", "BarbellRow"])
    parser.add_argument("--max_samples", default=None, type=int,
                        help="Cap clips/images per exercise")
    parser.add_argument("--dry_run",     action="store_true",
                        help="Verify paths only, no pose extraction")
    parser.add_argument("--workers",     default=None, type=int,
                        help=f"Workers (default: cpu_count-1 = {max(1,cpu_count()-1)})")
    parser.add_argument("--frame_skip",  default=FRAME_SKIP, type=int,
                        help=f"Every Nth frame (default: {FRAME_SKIP})")
    parser.add_argument("--frame_width", default=FRAME_WIDTH, type=int,
                        help=f"Resize width (default: {FRAME_WIDTH})")
    parser.add_argument("--complexity",  default=MODEL_COMPLEXITY, type=int,
                        choices=[0, 1, 2],
                        help=f"MediaPipe model complexity (default: {MODEL_COMPLEXITY})")
    args = parser.parse_args()

    exercises = list(EXERCISE_CONFIG) if args.exercise == "all" else [args.exercise]

    print(f"\n  Workers:     {args.workers or max(1, cpu_count()-1)}")
    print(f"  Frame skip:  every {args.frame_skip} frames")
    print(f"  Frame width: {args.frame_width}px")
    print(f"  Complexity:  {args.complexity}")

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
    print(f"  Total: {(time.time()-t0)/60:.1f} min")
    print(f"  Models → {Path(args.save_dir).resolve()}")
    print(f"{'═'*62}")