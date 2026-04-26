"""
Module 2: Repetition Segmentation
Tracks the hip midpoint vertical trajectory over time.
Uses scipy peak detection to identify individual rep boundaries.
"""

import numpy as np
from scipy.signal import find_peaks, savgol_filter
from dataclasses import dataclass, field
from typing import List


@dataclass
class Rep:
    start_frame: int
    bottom_frame: int
    end_frame: int
    rep_number: int


class RepSegmentor:
    def __init__(
        self,
        prominence: float = 0.02,   # lowered from 0.035 — catches shallower reps
        distance: int = 15,         # lowered from 20 — allows faster rep cadence
        smooth_window: int = 7,     # lowered from 11 — better for short clips
        smooth_poly: int = 3,
    ):
        """
        Args:
            prominence:    Min vertical hip displacement to count as a rep.
                           0.02 = 2% of frame height. Lower = more sensitive.
            distance:      Min frames between two rep peaks. 15 @ 30fps = 0.5s minimum.
            smooth_window: Savitzky-Golay filter window (must be odd, >= smooth_poly+2).
            smooth_poly:   Polynomial order for smoothing filter.
        """
        self.prominence = prominence
        self.distance = distance
        self.smooth_window = smooth_window
        self.smooth_poly = smooth_poly

        self.hip_y_buffer: List[float] = []
        self.confirmed_reps: List[Rep] = []

    def update(self, landmarks: np.ndarray):
        """Feed one landmark frame. Landmarks shape: (33, 4)."""
        hip_y = float((landmarks[23, 1] + landmarks[24, 1]) / 2.0)
        self.hip_y_buffer.append(hip_y)

    def _smooth_signal(self) -> np.ndarray:
        sig = np.array(self.hip_y_buffer)
        # Need at least smooth_window points; fall back to raw if too short
        if len(sig) < self.smooth_window:
            return sig
        return savgol_filter(sig, self.smooth_window, self.smooth_poly)

    def get_reps(self) -> List[Rep]:
        """
        Returns list of confirmed reps detected so far.
        In squat/row movements hip_y increases going DOWN (image y-axis).
        Troughs (local maxima of hip_y) = bottom of movement.
        Peaks  (local minima of hip_y) = top of movement.
        """
        sig = self._smooth_signal()

        # Need minimum frames to have any chance of a rep
        min_frames = max(30, self.distance * 2)
        if len(sig) < min_frames:
            return self.confirmed_reps

        # Troughs = bottom of squat (hip_y maximum)
        troughs, _ = find_peaks(sig,  prominence=self.prominence, distance=self.distance)
        # Peaks   = top of squat   (hip_y minimum)
        peaks,   _ = find_peaks(-sig, prominence=self.prominence, distance=self.distance)

        reps = []
        for trough in troughs:
            left_peaks  = peaks[peaks < trough]
            right_peaks = peaks[peaks > trough]
            if len(left_peaks) == 0 or len(right_peaks) == 0:
                continue
            start = int(left_peaks[-1])
            end   = int(right_peaks[0])
            # Exclude reps whose end boundary is within the last few frames
            # (the rep may still be in progress)
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
        """Returns 'descending', 'ascending', or 'top' from recent trajectory."""
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