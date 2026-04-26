"""
Module 4: Error Classification and Quality Scoring
One SVM per error type trained on Fitness-AQA feature vectors.
Quality score (0-100) derived from weighted error penalties.

Key fix for class imbalance:
  - SVC(class_weight='balanced') — auto-weights minority class
  - predict() uses probability threshold 0.3 instead of 0.5
    so the model is more sensitive to rare errors
"""

import numpy as np
import joblib
from pathlib import Path
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, Tuple, List


EXERCISE_ERRORS = {
    "BackSquat": {
        "knees_inward":      {"weight": 0.30, "feedback": "Keep your knees tracking over your toes."},
        "knees_forward":     {"weight": 0.15, "feedback": "Don't let knees travel too far past your toes."},
        "rounded_back":      {"weight": 0.35, "feedback": "Brace your core and maintain a neutral spine."},
        "shallow_squat":     {"weight": 0.20, "feedback": "Squat deeper — aim for thighs parallel or below."},
    },
    "OverheadPress": {
        "elbow_error":       {"weight": 0.45, "feedback": "Keep elbows at 45° and drive straight overhead."},
        "knees_error":       {"weight": 0.30, "feedback": "Keep knees soft and avoid hyperextension."},
        "spine_error":       {"weight": 0.25, "feedback": "Avoid excessive lumbar extension — brace the core."},
    },
    "BarbellRow": {
        "lumbar_error":      {"weight": 0.55, "feedback": "Maintain a flat back — avoid rounding the lumbar spine."},
        "torso_angle_error": {"weight": 0.45, "feedback": "Keep torso angle consistent — don't let hips rise early."},
    },
}

# Probability threshold for calling an error "detected"
# Lower than 0.5 makes the model more sensitive to minority (error) class
DETECTION_THRESHOLD = 0.35


class FormClassifier:
    def __init__(self, exercise: str = "BackSquat"):
        if exercise not in EXERCISE_ERRORS:
            raise ValueError(f"Unknown exercise '{exercise}'. "
                             f"Choose from {list(EXERCISE_ERRORS)}")
        self.exercise     = exercise
        self.error_config = EXERCISE_ERRORS[exercise]
        self.pipelines: Dict[str, dict] = {}

    def _make_pipeline(self) -> dict:
        return {
            "scaler": MinMaxScaler(),
            "svm": SVC(
                kernel="rbf",
                C=10.0,
                gamma="scale",
                probability=True,
                class_weight="balanced",   # ← auto-upweights minority error class
            ),
        }

    def train(self, X_train: np.ndarray, y_dict: Dict[str, np.ndarray]):
        for error_name, y in y_dict.items():
            if error_name not in self.error_config:
                print(f"[Classifier] Skipping '{error_name}' "
                      f"— not in config for {self.exercise}")
                continue
            classes, counts = np.unique(y, return_counts=True)
            if len(classes) < 2:
                print(f"[Classifier] Skipping '{error_name}' "
                      f"— only one class in training data "
                      f"(all {'positive' if classes[0]==1 else 'negative'})")
                continue
            pos_rate = counts[classes == 1][0] / len(y) if 1 in classes else 0
            print(f"[Classifier] Training '{error_name}' | "
                  f"{len(y)} samples | {pos_rate*100:.1f}% positive")
            pipe = self._make_pipeline()
            X_sc = pipe["scaler"].fit_transform(X_train)
            pipe["svm"].fit(X_sc, y)
            self.pipelines[error_name] = pipe
        print(f"[Classifier] Trained {len(self.pipelines)} "
              f"classifiers for {self.exercise}.")

    def predict(self, features: Dict[str, float],
                threshold: float = DETECTION_THRESHOLD
                ) -> Tuple[Dict[str, bool], int, List[str]]:
        x = np.array(list(features.values()), dtype=np.float32).reshape(1, -1)
        errors_detected: Dict[str, bool] = {}
        feedback_msgs:   List[str]       = []

        if not self.pipelines:
            errors_detected = self._heuristic_predict(features)
        else:
            for error_name, pipe in self.pipelines.items():
                X_sc = pipe["scaler"].transform(x)
                prob = pipe["svm"].predict_proba(X_sc)[0][1]
                errors_detected[error_name] = bool(prob >= threshold)

        total_penalty = 0.0
        for error_name, detected in errors_detected.items():
            if detected:
                cfg = self.error_config[error_name]
                total_penalty += cfg["weight"]
                feedback_msgs.append(cfg["feedback"])

        quality_score = max(0, int((1.0 - total_penalty) * 100))
        return errors_detected, quality_score, feedback_msgs

    def _heuristic_predict(self, features: Dict[str, float]) -> Dict[str, bool]:
        errors = {}
        if self.exercise == "BackSquat":
            errors["knees_inward"]  = features.get("knee_symmetry", 0) > 12.0
            errors["knees_forward"] = features.get("knee_angle_min", 180) < 60.0
            errors["rounded_back"]  = features.get("spine_angle_min", 180) < 50.0
            errors["shallow_squat"] = features.get("depth_range", 1.0) < 0.06
        elif self.exercise == "OverheadPress":
            errors["elbow_error"]   = features.get("elbow_symmetry", 0) > 15.0
            errors["knees_error"]   = features.get("knee_angle_min", 180) < 160.0
            errors["spine_error"]   = features.get("spine_angle_min", 180) < 45.0
        elif self.exercise == "BarbellRow":
            errors["lumbar_error"]      = features.get("spine_angle_min", 180) < 40.0
            errors["torso_angle_error"] = features.get("hip_angle_min", 180) < 30.0
        return errors

    def save(self, path: str):
        joblib.dump({
            "exercise":     self.exercise,
            "pipelines":    self.pipelines,
            "error_config": self.error_config,
        }, path)
        print(f"[Classifier] Saved → {path}")

    def load(self, path: str):
        data = joblib.load(path)
        self.exercise     = data["exercise"]
        self.pipelines    = data["pipelines"]
        self.error_config = data["error_config"]
        print(f"[Classifier] Loaded {self.exercise} from {path}")
        return self