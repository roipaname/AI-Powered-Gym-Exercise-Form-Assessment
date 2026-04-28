"""
main.py — Live Webcam Assessment  (FormIQ)
Run: python main.py --exercise BackSquat --model models/backsquat_svm.pkl
"""

import argparse
import cv2
import json
import math
import numpy as np
from pathlib import Path

from modules.extractor  import PoseExtractor
from modules.segmentor  import RepSegmentor
from modules.features   import compute_rep_features
from modules.classifier import FormClassifier
from modules.tracker    import SessionTracker

# ── Live feed sharing ─────────────────────────────────────────────────────────
# main.py writes frames + session state to .formiq_live/ so app.py can display them
import base64, json as _json
from pathlib import Path as _Path

LIVE_DIR   = _Path(".formiq_live")
FRAME_FILE = LIVE_DIR / "frame.jpg"
STATE_FILE = LIVE_DIR / "state.json"

def _write_live(frame_bgr, score, errors, reps, trend, fatigue, overload, phase):
    """Write current frame + state for app.py to read."""
    try:
        LIVE_DIR.mkdir(exist_ok=True)
        cv2.imwrite(str(FRAME_FILE), frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 70])
        _json.dumps({   # validate serialisable
            "score": score if isinstance(score, int) else 0,
            "errors": {k: bool(v) for k, v in errors.items()},
            "rep_count": reps,
            "trend": trend,
            "fatigue": fatigue,
            "overload": overload,
            "phase": phase,
            "ts": _json.dumps(None),   # placeholder
        })
        STATE_FILE.write_text(_json.dumps({
            "score": score if isinstance(score, int) else 0,
            "errors": {k: bool(v) for k, v in errors.items()},
            "rep_count": reps,
            "trend": trend,
            "fatigue": fatigue,
            "overload": overload,
            "phase": phase,
        }))
    except Exception:
        pass   # never crash the main loop over a write failure

def _clear_live():
    try:
        if FRAME_FILE.exists(): FRAME_FILE.unlink()
        if STATE_FILE.exists(): STATE_FILE.unlink()
    except Exception:
        pass


# ── Palette (BGR) ────────────────────────────────────────────────────────────
C_GREEN       = (80,  220, 100)
C_GREEN_BRIGHT= (80,  255, 140)
C_AMBER       = (40,  190, 250)
C_RED         = (60,   70, 240)
C_WHITE       = (230, 240, 240)
C_MUTED       = (120, 140, 130)
C_PANEL_BG    = (8,   22,  12)

SCORE_GOOD    = 75
SCORE_OK      = 50

MODEL_MAP = {
    "BackSquat":    "models/backsquat_svm.pkl",
    "OverheadPress":"models/overheadpress_svm.pkl",
    "BarbellRow":   "models/barbellrow_svm.pkl",
}


# ── Drawing helpers ───────────────────────────────────────────────────────────

def txt(frame, text, pos, scale=0.55, color=C_WHITE, thickness=1, bold=False):
    t = thickness + (1 if bold else 0)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, t, cv2.LINE_AA)


def score_color(s):
    if isinstance(s, int):
        if s >= SCORE_GOOD: return C_GREEN_BRIGHT
        if s >= SCORE_OK:   return C_AMBER
        return C_RED
    return C_MUTED


def draw_panel(frame, x, y, w, h, alpha=0.55):
    """Semi-transparent dark panel."""
    sub = frame[y:y+h, x:x+w]
    bg  = np.full_like(sub, C_PANEL_BG)
    cv2.addWeighted(bg, alpha, sub, 1-alpha, 0, sub)
    frame[y:y+h, x:x+w] = sub
    cv2.rectangle(frame, (x, y), (x+w, y+h),
                  (40, 100, 55), 1, cv2.LINE_AA)


def draw_score_ring(frame, cx, cy, radius, score):
    """Animated arc progress ring for quality score."""
    pct   = max(0, min(100, score if isinstance(score, int) else 0))
    angle = int(360 * pct / 100)
    col   = score_color(score)

    # Background ring
    cv2.ellipse(frame, (cx, cy), (radius, radius), -90, 0, 360,
                (30, 60, 35), 6, cv2.LINE_AA)
    # Foreground arc
    if angle > 0:
        cv2.ellipse(frame, (cx, cy), (radius, radius), -90, 0, angle,
                    col, 6, cv2.LINE_AA)
    # Score text
    label = str(score) if isinstance(score, int) else "--"
    fs    = 0.9 if isinstance(score, int) and score >= 10 else 1.0
    (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)
    txt(frame, label, (cx - tw//2, cy + 8), fs, col, 2, bold=True)
    txt(frame, "/100", (cx - 15, cy + 22), 0.32, C_MUTED)


def draw_mini_bar(frame, x, y, w, h, pct, color):
    """Filled progress bar."""
    cv2.rectangle(frame, (x, y), (x+w, y+h), (25, 55, 30), -1)
    filled = int(w * pct / 100)
    if filled > 0:
        cv2.rectangle(frame, (x, y), (x+filled, y+h), color, -1)
    cv2.rectangle(frame, (x, y), (x+w, y+h), (40, 80, 45), 1)


def draw_pill(frame, x, y, label, detected):
    """Rounded error pill indicator."""
    col    = C_RED if detected else C_GREEN
    symbol = "✗" if detected else "✓"
    full   = f"{symbol} {label.replace('_',' ')}"
    (tw, th), _ = cv2.getTextSize(full, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
    pad    = 6
    rx, ry = x, y - th - pad//2
    rw, rh = tw + pad*2, th + pad

    bg_col = (30, 25, 50) if detected else (15, 45, 20)
    cv2.rectangle(frame, (rx, ry), (rx+rw, ry+rh), bg_col, -1, cv2.LINE_AA)
    cv2.rectangle(frame, (rx, ry), (rx+rw, ry+rh), col, 1, cv2.LINE_AA)
    txt(frame, full, (rx+pad, y-2), 0.38, col)
    return rw + 6   # return width for layout


def draw_gradient_bar(frame, x, y, w, h):
    """Green→amber gradient title bar."""
    for i in range(w):
        t   = i / w
        r   = int(10  + t * 20)
        g   = int(180 + t * 30)
        b   = int(60  + t * 150)
        cv2.line(frame, (x+i, y), (x+i, y+h), (r, g, b), 1)


# ── Main ─────────────────────────────────────────────────────────────────────

def main(exercise="BackSquat", athlete_id="athlete",
         model_path=None, camera_id=0):

    extractor  = PoseExtractor()
    segmentor  = RepSegmentor()
    classifier = FormClassifier(exercise)
    tracker    = SessionTracker(exercise, athlete_id=athlete_id)

    # Auto-resolve model path
    if model_path is None:
        model_path = MODEL_MAP.get(exercise)
    if model_path and Path(model_path).exists():
        classifier.load(model_path)
        print(f"[FormIQ] Model loaded: {model_path}")
    else:
        print("[FormIQ] No model found — heuristic mode")

    # On macOS, camera needs AVFoundation backend explicitly
    cap = cv2.VideoCapture(camera_id, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        # Fallback to default backend
        cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"[FormIQ] Cannot open camera {camera_id}")
        print("  macOS: Grant camera access to Terminal in System Settings → Privacy → Camera")
        print("  Try: python main.py --camera 1  (if multiple cameras available)")
        return

    # Warmup — macOS camera needs a few frames before it streams
    for _ in range(5):
        cap.read()

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    PANEL_W = 300

    landmark_buffer = []
    last_rep_count  = 0
    current_score   = "--"
    current_errors  = {}
    current_feedback = []
    show_feedback_until = 0

    print(f"[FormIQ] Live — {exercise} | Press Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ── Pose extraction ───────────────────────────────────────────────
        rgb       = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        lm, ann   = extractor.extract(rgb)
        display   = cv2.cvtColor(ann, cv2.COLOR_RGB2BGR)

        if lm is not None:
            segmentor.update(lm)
            landmark_buffer.append(lm)
            reps = segmentor.get_reps()
            if len(reps) > last_rep_count:
                rep    = reps[-1]
                seq    = landmark_buffer[rep.start_frame: rep.end_frame + 1]
                if len(seq) > 3:
                    feats  = compute_rep_features(seq)
                    errors, score, feedback = classifier.predict(feats)
                    tracker.log_rep(score, errors, feedback,
                                    segmentor.current_phase())
                    current_score    = score
                    current_errors   = errors
                    current_feedback = feedback
                    show_feedback_until = cv2.getTickCount() + cv2.getTickFrequency() * 4
                last_rep_count = len(reps)

        import time
        show_fb = cv2.getTickCount() < show_feedback_until

        # ── Left panel ────────────────────────────────────────────────────
        draw_panel(display, 0, 0, PANEL_W, H)

        # Gradient title strip
        draw_gradient_bar(display, 0, 0, PANEL_W, 36)
        txt(display, "FORMIQ", (10, 24), 0.65, (10, 20, 10), 2, bold=True)
        txt(display, exercise.upper(), (80, 24), 0.42, (10, 30, 10))

        # Score ring
        draw_score_ring(display, 60, 100, 42, current_score)

        # Rep counter
        rep_count = len(tracker.reps)
        txt(display, "REPS", (125, 75), 0.35, C_MUTED)
        txt(display, str(rep_count), (125, 105), 1.1, C_WHITE, 2, bold=True)

        # Phase indicator
        phase = segmentor.current_phase()
        phase_col = C_GREEN if phase == "descending" else \
                    C_AMBER if phase == "ascending"  else C_MUTED
        txt(display, f"● {phase.upper()}", (125, 125), 0.38, phase_col)

        # Divider
        cv2.line(display, (10, 148), (PANEL_W-10, 148), (40, 80, 50), 1)

        # Error pills
        y_err = 168
        if current_errors:
            txt(display, "FORM ERRORS", (10, y_err - 4), 0.32, C_MUTED)
            y_err += 14
            x_pill = 10
            for err, detected in current_errors.items():
                pw = draw_pill(display, x_pill, y_err, err, detected)
                y_err += 22

        # Divider
        cv2.line(display, (10, y_err + 4), (PANEL_W-10, y_err+4),
                 (40, 80, 50), 1)
        y_err += 16

        # Trend + session stats
        trend_label, slope = tracker.trend()
        trend_col = C_GREEN if trend_label == "improving" else \
                    C_RED   if trend_label == "declining"  else C_MUTED
        txt(display, "TREND", (10, y_err), 0.32, C_MUTED)
        txt(display, f"{trend_label.upper()}", (10, y_err + 16),
            0.42, trend_col, bold=True)
        if len(tracker.reps) >= 4:
            txt(display, f"{slope:+.1f}/rep", (120, y_err + 16),
                0.35, C_MUTED)

        # Avg score bar
        scores = tracker.scores
        if scores:
            avg = int(np.mean(scores))
            y_bar = y_err + 36
            txt(display, f"AVG  {avg}", (10, y_bar), 0.38, C_MUTED)
            draw_mini_bar(display, 10, y_bar + 6, PANEL_W - 20, 5,
                          avg, score_color(avg))

        # Alerts at bottom of panel
        alert_y = H - 80
        if tracker.fatigue_alert():
            draw_panel(display, 5, alert_y, PANEL_W-10, 28, 0.7)
            cv2.rectangle(display, (5, alert_y), (PANEL_W-5, alert_y+28),
                          C_RED, 1, cv2.LINE_AA)
            txt(display, "⚡ FATIGUE — CONSIDER REST",
                (12, alert_y + 18), 0.38, C_RED, bold=True)
            alert_y -= 34

        if tracker.overload_ready():
            draw_panel(display, 5, alert_y, PANEL_W-10, 28, 0.7)
            cv2.rectangle(display, (5, alert_y), (PANEL_W-5, alert_y+28),
                          C_GREEN_BRIGHT, 1, cv2.LINE_AA)
            txt(display, "🚀 READY TO INCREASE LOAD",
                (12, alert_y + 18), 0.38, C_GREEN_BRIGHT, bold=True)

        # ── Feedback overlay (fades after 4s) ─────────────────────────────
        if show_fb and current_feedback:
            fb_x = PANEL_W + 10
            fb_y = H - len(current_feedback) * 26 - 16
            for fb in current_feedback:
                draw_panel(display, fb_x, fb_y - 18,
                           W - PANEL_W - 20, 22, 0.65)
                txt(display, f"→  {fb}", (fb_x + 8, fb_y),
                    0.42, C_AMBER)
                fb_y += 26

        # ── Right-edge score flash (new rep) ──────────────────────────────
        import time as _t
        if show_fb and isinstance(current_score, int):
            flash_col = score_color(current_score)
            cv2.rectangle(display, (W-5, 0), (W, H), flash_col, -1)

        # ── Quit hint ─────────────────────────────────────────────────────
        txt(display, "Q to quit", (W - 75, H - 10), 0.32, C_MUTED)

        # Share frame + state with app.py
        trend_label, slope = tracker.trend()
        _write_live(
            display,
            current_score,
            current_errors,
            len(tracker.reps),
            trend_label,
            tracker.fatigue_alert(),
            tracker.overload_ready(),
            segmentor.current_phase(),
        )

        cv2.imshow("FormIQ — Live Assessment", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    extractor.close()
    _clear_live()

    summary = tracker.summary()
    saved   = tracker.save()
    print("\n── Session Summary ──────────────────────────────────────")
    print(json.dumps(
        {k: v for k, v in summary.items() if k != "rep_log"},
        indent=2))
    print(f"Log saved → {saved}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FormIQ — Live Exercise Assessment")
    parser.add_argument("--exercise", default="BackSquat",
                        choices=["BackSquat", "OverheadPress", "BarbellRow"])
    parser.add_argument("--athlete",  default="athlete")
    parser.add_argument("--model",    default=None,
                        help="Path to .pkl (auto-resolved if omitted)")
    parser.add_argument("--camera",   default=0, type=int)
    args = parser.parse_args()
    main(args.exercise, args.athlete, args.model, args.camera)