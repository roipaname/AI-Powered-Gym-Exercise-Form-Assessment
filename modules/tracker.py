"""
Module 5 (NEW): Progressive Overload & Session Tracker
Logs rep-by-rep quality scores, detects form-degradation trends,
and generates progressive overload readiness signals.
"""

import json
import datetime
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field, asdict   
#light changes

#light changes


@dataclass
class RepRecord:
    rep_number:     int
    score:          int
    errors:         Dict[str, bool]
    feedback:       List[str]
    phase_at_bottom: str = "unknown"
    timestamp:      str = field(default_factory=lambda: datetime.datetime.now().isoformat())


class SessionTracker:
    """
    Tracks form quality across an entire workout session.
    Provides:
      - Running quality trend (linear regression slope)
      - Fatigue alert when form degrades significantly
      - Progressive overload readiness signal
      - Session summary JSON export
    """

    OVERLOAD_THRESHOLD      = 80    # min score for last N reps to flag "ready"
    OVERLOAD_WINDOW         = 3     # how many consecutive good reps needed
    FATIGUE_SLOPE_THRESHOLD = -3.0  # score/rep slope below which to warn
    PLATEAU_WINDOW          = 5     # reps to detect score plateau

    def __init__(self, exercise: str, athlete_id: str = "athlete", save_dir: str = "logs"):
        self.exercise   = exercise
        self.athlete_id = athlete_id
        self.save_dir   = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.reps: List[RepRecord] = []
        self.session_start = datetime.datetime.now()
        self._session_id   = self.session_start.strftime("%Y%m%d_%H%M%S")

    # ── Ingestion ──────────────────────────────────────────────────

    def log_rep(self, score: int, errors: Dict[str, bool],
                feedback: List[str], phase: str = "unknown"):
        record = RepRecord(
            rep_number=len(self.reps) + 1,
            score=score,
            errors=errors,
            feedback=feedback,
            phase_at_bottom=phase,
        )
        self.reps.append(record)

    # ── Analysis ───────────────────────────────────────────────────

    @property
    def scores(self) -> List[int]:
        return [r.score for r in self.reps]

    def trend(self) -> Tuple[str, float]:
        """
        Returns (label, slope_per_rep).
        Needs at least 4 reps for a meaningful trend.
        """
        s = self.scores
        if len(s) < 4:
            return "insufficient data", 0.0
        slope = float(np.polyfit(range(len(s)), s, 1)[0])
        if slope < self.FATIGUE_SLOPE_THRESHOLD:
            label = "declining"
        elif slope > 2.0:
            label = "improving"
        else:
            label = "stable"
        return label, slope

    def fatigue_alert(self) -> bool:
        """True when form is declining fast — recommend stopping or reducing load."""
        label, slope = self.trend()
        return label == "declining"

    def overload_ready(self) -> bool:
        """True when last N reps all exceeded the quality threshold."""
        s = self.scores
        if len(s) < self.OVERLOAD_WINDOW:
            return False
        return all(score >= self.OVERLOAD_THRESHOLD for score in s[-self.OVERLOAD_WINDOW:])

    def plateau_detected(self) -> bool:
        """True when recent scores are clustered within ±5 points (stagnation)."""
        s = self.scores
        if len(s) < self.PLATEAU_WINDOW:
            return False
        recent = s[-self.PLATEAU_WINDOW:]
        return (max(recent) - min(recent)) <= 5

    def error_frequency(self) -> Dict[str, int]:
        """How many times each error appeared across all reps."""
        freq: Dict[str, int] = {}
        for rep in self.reps:
            for err, detected in rep.errors.items():
                if detected:
                    freq[err] = freq.get(err, 0) + 1
        return dict(sorted(freq.items(), key=lambda x: -x[1]))

    def running_average(self, window: int = 3) -> List[float]:
        """Smoothed moving average of scores."""
        s = np.array(self.scores, dtype=float)
        if len(s) < window:
            return list(s)
        return [float(np.mean(s[max(0, i - window + 1):i + 1])) for i in range(len(s))]

    def coaching_tip(self) -> str:
        """Top corrective tip based on the most frequent error."""
        freq = self.error_frequency()
        if not freq:
            return "Great session — no consistent errors detected."
        top_error = next(iter(freq))
        # Map to a generic tip if no classifier feedback is accessible
        tips = {
            "knees_inward":  "Focus drill: 3 sets of banded squats to cue knee tracking.",
            "rounded_back":  "Focus drill: Paused Romanian deadlift to reinforce neutral spine.",
            "shallow_squat": "Focus drill: Box squat to a low box — build depth confidence.",
            "knees_forward": "Focus drill: Wall squat drill — toes close to wall.",
            "elbow_error":   "Focus drill: Z-press seated to isolate vertical pressing path.",
            "lumbar_error":  "Focus drill: Chest-supported row to remove spinal loading.",
            "asymmetry":     "Focus: Single-arm work to correct left-right imbalance.",
        }
        return tips.get(top_error, "Review your form with a qualified coach.")

    # ── Export ─────────────────────────────────────────────────────

    def summary(self) -> Dict:
        s = self.scores
        trend_label, slope = self.trend()
        duration = (datetime.datetime.now() - self.session_start).seconds

        return {
            "session_id":       self._session_id,
            "athlete_id":       self.athlete_id,
            "exercise":         self.exercise,
            "date":             self.session_start.strftime("%Y-%m-%d"),
            "duration_seconds": duration,
            "total_reps":       len(self.reps),
            "avg_score":        round(float(np.mean(s)), 1) if s else 0.0,
            "best_score":       max(s) if s else 0,
            "worst_score":      min(s) if s else 0,
            "trend":            trend_label,
            "slope_per_rep":    round(slope, 3),
            "fatigue_alert":    self.fatigue_alert(),
            "overload_ready":   self.overload_ready(),
            "plateau_detected": self.plateau_detected(),
            "error_frequency":  self.error_frequency(),
            "top_coaching_tip": self.coaching_tip(),
            "rep_log":          [asdict(r) for r in self.reps],
        }

    def save(self) -> Path:
        data = self.summary()
        fname = self.save_dir / f"{self.exercise}_{self._session_id}.json"
        with open(fname, "w") as f:
            json.dump(data, f, indent=2)
        return fname

    def reset(self):
        self.reps = []
        self.session_start = datetime.datetime.now()
        self._session_id   = self.session_start.strftime("%Y%m%d_%H%M%S")
