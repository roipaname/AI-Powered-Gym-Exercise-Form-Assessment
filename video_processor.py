

import os
os.environ["GLOG_minloglevel"]      = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["MEDIAPIPE_DISABLE_GPU"]= "1"
import warnings; warnings.filterwarnings("ignore")

import cv2
import numpy as np
import threading
import datetime
import streamlit as st

try:
    import av
    from streamlit_webrtc import VideoProcessorBase
    WEBRTC_OK = True
except ImportError:
    WEBRTC_OK = False
    VideoProcessorBase = object

from modules.extractor  import PoseExtractor
from modules.segmentor  import RepSegmentor
from modules.features   import compute_rep_features
from modules.classifier import FormClassifier, EXERCISE_ERRORS
from modules.tracker    import SessionTracker


# ── BGR colour constants ───────────────────────────────────────────────────────
C_LIME  = (62,  255, 168)
C_TEAL  = (160, 230,  0)
C_AMBER = (0,   184, 255)
C_RED   = (109,  77, 255)
C_WHITE = (235, 245, 232)
C_MUTED = (100, 130, 110)
C_DARK  = (8,   20,  10)

MODEL_MAP = {
    "BackSquat":    "models/backsquat_svm.pkl",
    "OverheadPress":"models/overheadpress_svm.pkl",
    "BarbellRow":   "models/barbellrow_svm.pkl",
}


def score_color(s):
    if isinstance(s, int):
        if s >= 75: return C_LIME
        if s >= 50: return C_AMBER
    return C_RED


def draw_hud(frame: np.ndarray, score, errors: dict, feedback: list,
             rep_count: int, phase: str, trend: str, fatigue: bool, overload: bool):
    """
    Draw the full HUD overlay onto the frame in-place.
    Panel is 25% of frame width, semi-transparent dark green.
    """
    h, w = frame.shape[:2]
    pw   = max(200, w // 4)

    # ── Semi-transparent panel ─────────────────────────────────────────────
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (pw, h), C_DARK, -1)
    cv2.addWeighted(overlay, 0.58, frame, 0.42, 0, frame)
    cv2.rectangle(frame, (0, 0), (pw, h), (55, 160, 45), 1, cv2.LINE_AA)

    # Gradient title bar
    for i in range(pw):
        t = i / pw
        cv2.line(frame, (i, 0), (i, 32),
                 (int(3+t*8), int(180+t*75), int(60+t*108)), 1)
    cv2.putText(frame, "FORMIQ", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (5, 12, 5), 2, cv2.LINE_AA)

    # ── Score arc ring ─────────────────────────────────────────────────────
    cx, cy, r = pw // 2 - 20, 90, 36
    sc  = score if isinstance(score, int) else 0
    col = score_color(score)
    cv2.ellipse(frame, (cx, cy), (r, r), -90, 0, 360, (20, 60, 20), 5, cv2.LINE_AA)
    if sc > 0:
        cv2.ellipse(frame, (cx, cy), (r, r), -90, 0, int(360*sc/100), col, 5, cv2.LINE_AA)
    lbl = str(sc) if isinstance(score, int) else "--"
    tw  = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0][0]
    cv2.putText(frame, lbl, (cx-tw//2, cy+7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2, cv2.LINE_AA)
    cv2.putText(frame, "/100", (cx-12, cy+20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.28, C_MUTED, 1, cv2.LINE_AA)

    # ── Rep counter ────────────────────────────────────────────────────────
    rx = cx + r + 14
    cv2.putText(frame, "REPS", (rx, cy-14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, C_MUTED, 1, cv2.LINE_AA)
    cv2.putText(frame, str(rep_count), (rx, cy+12),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, C_WHITE, 2, cv2.LINE_AA)

    # Phase
    ph_col = C_LIME if phase == "descending" else C_AMBER if phase == "ascending" else C_MUTED
    cv2.putText(frame, phase.upper(), (rx, cy+26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, ph_col, 1, cv2.LINE_AA)

    # ── Divider ────────────────────────────────────────────────────────────
    y = 138
    cv2.line(frame, (6, y), (pw-6, y), (40, 100, 30), 1)
    y += 14

    # ── Error pills ────────────────────────────────────────────────────────
    if errors:
        cv2.putText(frame, "FORM ERRORS", (6, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, C_MUTED, 1, cv2.LINE_AA)
        y += 14
        for err, det in errors.items():
            c = C_RED if det else C_LIME
            sym = "X" if det else "OK"
            cv2.putText(frame, f"{sym} {err.replace('_',' ')}", (6, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, c, 1, cv2.LINE_AA)
            y += 18
    y += 4
    cv2.line(frame, (6, y), (pw-6, y), (40, 100, 30), 1)
    y += 12

    # ── Feedback text (word-wrapped) ───────────────────────────────────────
    for fb in feedback[:2]:
        words = fb.split()
        line  = ""
        for wd in words:
            test = (line + " " + wd).strip()
            if cv2.getTextSize(test, cv2.FONT_HERSHEY_SIMPLEX, 0.29, 1)[0][0] < pw - 12:
                line = test
            else:
                cv2.putText(frame, line, (6, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.29, C_AMBER, 1, cv2.LINE_AA)
                y += 13
                line = wd
        if line:
            cv2.putText(frame, line, (6, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.29, C_AMBER, 1, cv2.LINE_AA)
            y += 13

    # ── Bottom alerts ──────────────────────────────────────────────────────
    alert_y = h - 16
    if overload:
        cv2.putText(frame, "READY TO INCREASE LOAD", (6, alert_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, C_LIME, 1, cv2.LINE_AA)
        alert_y -= 18
    if fatigue:
        cv2.putText(frame, "! FATIGUE - REST", (6, alert_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_RED, 1, cv2.LINE_AA)
        alert_y -= 18

    # Trend
    tr_col = C_LIME if trend == "improving" else C_RED if trend == "declining" else C_MUTED
    cv2.putText(frame, f"TREND: {trend.upper()}", (6, alert_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.28, tr_col, 1, cv2.LINE_AA)

    # ── Right edge score flash ─────────────────────────────────────────────
    cv2.rectangle(frame, (w-4, 0), (w, h), score_color(score), -1)


class FormVideoProcessor(VideoProcessorBase if WEBRTC_OK else object):
    """
    WebRTC video processor — runs the full AI pipeline on each frame.
    Thread-safe: recv() runs in a background thread.
    Session state is written via a deque callback to avoid thread conflicts.
    """

    def __init__(self, exercise: str = "BackSquat"):
        self.exercise   = exercise
        self.extractor  = PoseExtractor(
            min_detection_conf=0.4,
            min_tracking_conf=0.4,
            model_complexity=1,
        )
        self.segmentor  = RepSegmentor()
        self.tracker    = SessionTracker(exercise)
        self.classifier = FormClassifier(exercise)

        mp = MODEL_MAP.get(exercise)
        from pathlib import Path
        if mp and Path(mp).exists():
            self.classifier.load(mp)

        self.lm_buf          = []
        self.last_rep_count  = 0
        self.lock            = threading.Lock()
        self._score          = 0
        self._errors         = {}
        self._feedback       = []
        self._rep_count      = 0
        self._phase          = "top"
        self._trend          = "insufficient data"
        self._fatigue        = False
        self._overload       = False
        # Queue of new rep dicts to push to session_state (written in recv, read in main thread)
        self._new_reps       = []

    @property
    def rep_count(self): return self._rep_count
    @property
    def score(self):     return self._score
    @property
    def errors(self):    return self._errors
    @property
    def feedback(self):  return self._feedback

    def recv(self, frame: "av.VideoFrame") -> "av.VideoFrame":
        img  = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]
        rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        lm, ann = self.extractor.extract(rgb)
        display  = cv2.cvtColor(ann, cv2.COLOR_RGB2BGR)

        if lm is not None:
            self.segmentor.update(lm)
            self.lm_buf.append(lm)

            reps = self.segmentor.get_reps()

            if len(reps) > self.last_rep_count:
                rep = reps[-1]
                seq = self.lm_buf[rep.start_frame: rep.end_frame + 1]
                if len(seq) >= 2:
                    try:
                        feats                  = compute_rep_features(seq)
                        errors, score, feedback = self.classifier.predict(feats)
                        self.tracker.log_rep(score, errors, feedback,
                                             self.segmentor.current_phase())
                        trend_label, _         = self.tracker.trend()
                        with self.lock:
                            self._score    = score
                            self._errors   = errors
                            self._feedback = feedback
                            self._rep_count = len(reps)
                            self._trend    = trend_label
                            self._fatigue  = self.tracker.fatigue_alert()
                            self._overload = self.tracker.overload_ready()
                            self._new_reps.append({
                                "rep_number": len(reps),
                                "score":      score,
                                "errors":     {k: bool(v) for k, v in errors.items()},
                                "feedback":   feedback,
                                "timestamp":  datetime.datetime.now().isoformat(),
                            })
                    except Exception:
                        pass
                self.last_rep_count = len(reps)

        with self.lock:
            self._phase = self.segmentor.current_phase()
            draw_hud(display,
                     self._score, self._errors, self._feedback,
                     self._rep_count, self._phase,
                     self._trend, self._fatigue, self._overload)

        return av.VideoFrame.from_ndarray(display, format="bgr24")

    def flush_new_reps(self):
        """Call from main Streamlit thread to safely collect new reps."""
        with self.lock:
            reps = list(self._new_reps)
            self._new_reps.clear()
        return reps

    def __del__(self):
        try:
            self.extractor.close()
        except Exception:
            pass
