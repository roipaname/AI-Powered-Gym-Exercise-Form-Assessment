"""
Module 3: Biomechanical Feature Computation
Computes per-repetition feature vectors from landmark sequences.
Features: joint angles, symmetry, depth, temporal smoothness.
"""

import numpy as np
from typing import List, Dict


def joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Compute the interior angle (degrees) at joint B, formed by A-B-C.
    a, b, c are 3D coordinate arrays [x, y, z].
    """
    ba = a - b
    bc = c - b
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba < 1e-8 or norm_bc < 1e-8:
        return 0.0
    cos_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    return float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))


def _safe_diff_var(arr: np.ndarray) -> float:
    """Variance of np.diff — returns 0.0 if array too short to diff."""
    if len(arr) < 2:
        return 0.0
    return float(np.var(np.diff(arr)))


def _safe_diff_mean(arr: np.ndarray) -> float:
    """Mean of abs(np.diff) — returns 0.0 if array too short to diff."""
    if len(arr) < 2:
        return 0.0
    return float(np.mean(np.abs(np.diff(arr))))


def compute_rep_features(landmark_sequence: List[np.ndarray]) -> Dict[str, float]:
    """
    Args:
        landmark_sequence: List of (33, 4) landmark arrays for one complete rep.
                           Minimum 2 frames required; single-frame seqs are safe.

    Returns:
        Dictionary of named biomechanical features (always 16 keys).
    """
    lms = np.array(landmark_sequence)   # (T, 33, 4)
    T   = lms.shape[0]
    xyz = lms[:, :, :3]                 # (T, 33, 3)

    # ── Joint angle time series ───────────────────────────────────────────────
    knee_L  = np.array([joint_angle(xyz[t, 23], xyz[t, 25], xyz[t, 27]) for t in range(T)])
    knee_R  = np.array([joint_angle(xyz[t, 24], xyz[t, 26], xyz[t, 28]) for t in range(T)])
    hip_L   = np.array([joint_angle(xyz[t, 11], xyz[t, 23], xyz[t, 25]) for t in range(T)])
    hip_R   = np.array([joint_angle(xyz[t, 12], xyz[t, 24], xyz[t, 26]) for t in range(T)])
    elbow_L = np.array([joint_angle(xyz[t, 11], xyz[t, 13], xyz[t, 15]) for t in range(T)])
    elbow_R = np.array([joint_angle(xyz[t, 12], xyz[t, 14], xyz[t, 16]) for t in range(T)])
    spine   = np.array([joint_angle(xyz[t, 11], xyz[t, 23], xyz[t, 25]) for t in range(T)])

    # Hip midpoint vertical displacement (image y increases downward)
    hip_y = (xyz[:, 23, 1] + xyz[:, 24, 1]) / 2.0

    # ── Feature dict (always 16 entries, safe for T=1) ────────────────────────
    features = {
        # Knee
        "knee_angle_min":    float(min(knee_L.min(), knee_R.min())),
        "knee_angle_max":    float(max(knee_L.max(), knee_R.max())),
        "knee_angle_range":  float(max(knee_L.max() - knee_L.min(),
                                       knee_R.max() - knee_R.min())),
        # Hip
        "hip_angle_min":     float(min(hip_L.min(), hip_R.min())),
        "hip_angle_max":     float(max(hip_L.max(), hip_R.max())),

        # Elbow (OHP primary, also useful for row symmetry)
        "elbow_angle_min":   float(min(elbow_L.min(), elbow_R.min())),
        "elbow_angle_max":   float(max(elbow_L.max(), elbow_R.max())),

        # Spine
        "spine_angle_min":   float(spine.min()),
        "spine_angle_mean":  float(spine.mean()),

        # Symmetry — mean absolute left-right difference
        "knee_symmetry":     float(np.mean(np.abs(knee_L - knee_R))),
        "hip_symmetry":      float(np.mean(np.abs(hip_L  - hip_R))),
        "elbow_symmetry":    float(np.mean(np.abs(elbow_L - elbow_R))),

        # Depth of movement
        "depth_range":       float(hip_y.max() - hip_y.min()),

        # Temporal smoothness — safe for T=1 (returns 0.0)
        "knee_temporal_var": _safe_diff_var(knee_L),
        "hip_temporal_var":  _safe_diff_var(hip_L),

        # Average angular velocity
        "knee_avg_velocity": _safe_diff_mean(knee_L),
    }

    return features


def feature_vector(features: Dict[str, float]) -> np.ndarray:
    """Convert feature dict to ordered numpy array for model input."""
    return np.array([features[k] for k in FEATURE_KEYS], dtype=np.float32)


FEATURE_KEYS = [
    "knee_angle_min", "knee_angle_max", "knee_angle_range",
    "hip_angle_min",  "hip_angle_max",
    "elbow_angle_min", "elbow_angle_max",
    "spine_angle_min", "spine_angle_mean",
    "knee_symmetry",  "hip_symmetry",  "elbow_symmetry",
    "depth_range",
    "knee_temporal_var", "hip_temporal_var",
    "knee_avg_velocity",
]