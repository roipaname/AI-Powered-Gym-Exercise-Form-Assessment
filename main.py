"""
main.py — CLI entry point for live webcam assessment.
Run: python main.py --exercise BackSquat
"""

import argparse
import cv2
import json
import numpy as np
from pathlib import Path

from modules.extractor  import PoseExtractor
from modules.segmentor  import RepSegmentor
from modules.features   import compute_rep_features
from modules.classifier import FormClassifier
from modules.tracker    import SessionTracker


def put_text(frame, text, pos, scale=0.65, color=(0, 255, 100), thickness=2):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def main(exercise: str = "BackSquat", athlete_id: str = "athlete",
         model_path: str = None, camera_id: int = 0):

    extractor  = PoseExtractor()
    segmentor  = RepSegmentor()
    classifier = FormClassifier(exercise)
    tracker    = SessionTracker(exercise, athlete_id=athlete_id)

    if model_path and Path(model_path).exists():
        classifier.load(model_path)
        print(f"[Main] Loaded classifier from {model_path}")
    else:
        print("[Main] No model found — running in heuristic demo mode.")

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"[Main] Cannot open camera {camera_id}. Exiting.")
        return

    landmark_buffer = []
    last_rep_count  = 0
    current_score   = "--"
    current_errors  = {}
    current_feedback = []

    print(f"[Main] Starting live assessment | Exercise: {exercise} | Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        lm, annotated_rgb = extractor.extract(frame_rgb)
        display = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)

        if lm is not None:
            segmentor.update(lm)
            landmark_buffer.append(lm)

            reps = segmentor.get_reps()
            if len(reps) > last_rep_count:
                new_rep = reps[-1]
                seq = landmark_buffer[new_rep.start_frame: new_rep.end_frame + 1]
                if len(seq) > 5:
                    features = compute_rep_features(seq)
                    errors, score, feedback = classifier.predict(features)
                    tracker.log_rep(score, errors, feedback, segmentor.current_phase())
                    current_score   = score
                    current_errors  = errors
                    current_feedback = feedback
                last_rep_count = len(reps)

        # ── HUD Overlay ─────────────────────────────────────────
        h, w = display.shape[:2]
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (320, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.35, display, 0.65, 0, display)

        put_text(display, f"Exercise: {exercise}", (10, 30), 0.55, (180, 255, 160))
        put_text(display, f"Reps: {len(tracker.reps)}", (10, 60), 0.7, (255, 255, 255))
        put_text(display, f"Score: {current_score}", (10, 95), 0.9,
                 (0, 220, 80) if (isinstance(current_score, int) and current_score >= 75)
                 else (0, 140, 255))

        phase = segmentor.current_phase()
        put_text(display, f"Phase: {phase}", (10, 125), 0.55, (200, 200, 200))

        y = 160
        for err, detected in current_errors.items():
            color = (0, 60, 255) if detected else (0, 200, 80)
            symbol = "X" if detected else "OK"
            put_text(display, f"[{symbol}] {err}", (10, y), 0.45, color)
            y += 22

        if tracker.fatigue_alert():
            put_text(display, "! FATIGUE ALERT — REST", (10, h - 70), 0.65, (0, 0, 255))
        if tracker.overload_ready():
            put_text(display, "READY TO INCREASE LOAD", (10, h - 45), 0.65, (0, 255, 150))

        trend_label, slope = tracker.trend()
        put_text(display, f"Trend: {trend_label} ({slope:+.1f}/rep)", (10, h - 20), 0.5, (180, 255, 160))

        cv2.imshow("Gym AQ System", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    extractor.close()

    summary = tracker.summary()
    saved = tracker.save()
    print("\n── Session Summary ──────────────────────────────")
    print(json.dumps({k: v for k, v in summary.items() if k != "rep_log"}, indent=2))
    print(f"Full log saved to: {saved}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gym Exercise Form Assessment — Live Mode")
    parser.add_argument("--exercise",   default="BackSquat",
                        choices=["BackSquat", "OverheadPress", "BarbellRow"])
    parser.add_argument("--athlete",    default="athlete", help="Athlete ID for logging")
    parser.add_argument("--model",      default=None, help="Path to .pkl classifier file")
    parser.add_argument("--camera",     default=0, type=int, help="Camera device index")
    args = parser.parse_args()
    main(args.exercise, args.athlete, args.model, args.camera)
