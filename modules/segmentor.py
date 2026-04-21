"""
Module 2: Repetition Segmentation
Tracks the hip midpoint vertical trajectory over time.
Uses scipy peak detection to identify individual rep boundaries.
"""

import numpy as np
from scipy.signal import find_peaks, savgol_filter
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class Rep:
    start_frame: int
    bottom_frame: int
    end_frame: int
    rep_number: int


class RepSegmentor:
    def __init__(
        self,
        prominence: float = 0.035,
        distance: int = 20,
        smooth_window: int = 11,
        smooth_poly: int = 3,
    ):
        """
        Args:
            prominence: Minimum depth ratio for a valid rep (fraction of frame height).
            distance:   Minimum frames between reps.
            smooth_window: Savitzky-Golay filter window for noise removal.
            smooth_poly:   Polynomial order for smoothing.
        """
        self.prominence = prominence
        self.distance = distance
        self.smooth_window = smooth_window
        self.smooth_poly = smooth_poly

        self.hip_y_buffer: List[float] = []
        self.confirmed_reps: List[Rep] = []
        self._last_processed_count = 0

    def update(self, landmarks: np.ndarray):
        """Feed a new landmark frame. Landmarks shape: (33, 4)."""
        hip_y = float((landmarks[23, 1] + landmarks[24, 1]) / 2.0)
        self.hip_y_buffer.append(hip_y)

    def _smooth_signal(self) -> np.ndarray:
        sig = np.array(self.hip_y_buffer)
        if len(sig) < self.smooth_window:
            return sig
        return savgol_filter(sig, self.smooth_window, self.smooth_poly)

    def get_reps(self) -> List[Rep]:
        """
        Returns the list of fully confirmed reps detected so far.
        In squat/hip-hinge movements, hip_y increases going DOWN (image coords).
        Troughs = bottom of movement; peaks = top.
        """
        sig = self._smooth_signal()
        if len(sig) < 30:
            return self.confirmed_reps

        # Detect troughs (bottom of rep = local maxima in image-y)
        troughs, _ = find_peaks(sig, prominence=self.prominence, distance=self.distance)
        # Detect peaks (top of rep = local minima)
        peaks, _ = find_peaks(-sig, prominence=self.prominence, distance=self.distance)

        reps = []
        for trough in troughs:
            left_peaks  = peaks[peaks < trough]
            right_peaks = peaks[peaks > trough]
            if len(left_peaks) == 0 or len(right_peaks) == 0:
                continue
            start = int(left_peaks[-1])
            end   = int(right_peaks[0])
            # Only include if the right boundary is not the last few frames (rep still in progress)
            if end < len(sig) - self.distance // 2:
                reps.append(Rep(
                    start_frame=start,
                    bottom_frame=int(trough),
                    end_frame=end,
                    rep_number=len(reps) + 1,
                ))

        self.confirmed_reps = reps
        return reps

    def current_phase(self) -> str:
        """Returns 'descending', 'ascending', or 'top' based on recent trajectory."""
        if len(self.hip_y_buffer) < 10:
            return "top"
        recent = np.array(self.hip_y_buffer[-10:])
        slope = np.polyfit(range(10), recent, 1)[0]
        if slope > 0.002:
            return "descending"
        elif slope < -0.002:
            return "ascending"
        return "top"

    def reset(self):
        self.hip_y_buffer = []
        self.confirmed_reps = []
        self._last_processed_count = 0
