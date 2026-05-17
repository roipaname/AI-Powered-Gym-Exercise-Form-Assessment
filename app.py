"""
app.py — FormIQ Streamlit Dashboard
=====================================
Run:  streamlit run app.py

Dependencies for live camera:
  pip install streamlit-webrtc aiortc av

Imports from:
  theme.py           — CSS + HTML component builders
  video_processor.py — WebRTC AI pipeline
  modules/           — extractor, segmentor, features, classifier, tracker
"""

# ── Suppress MediaPipe logs ────────────────────────────────────────────────────
import os
os.environ["GLOG_minloglevel"]      = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["MEDIAPIPE_DISABLE_GPU"]= "1"
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import numpy as np
import json
import time
import random
import datetime
from pathlib import Path

# ── WebRTC (optional) ─────────────────────────────────────────────────────────
try:
    from streamlit_webrtc import webrtc_streamer, RTCConfiguration
    from video_processor import FormVideoProcessor
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False

# ── Design system ─────────────────────────────────────────────────────────────
from theme import (
    CSS, card, section_label, score_ring_svg, pill,
    trend_chip, mini_bar, alert_box, status_dot,
)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG  (must be first Streamlit call)
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="FormIQ — AI Gym Coach",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(CSS, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
def init_state():
    defaults = {
        "reps":          [],
        "exercise":      "BackSquat",
        "athlete_id":    "Athlete",
        "session_start": None,
        "live_score":    0,
        "live_errors":   {},
        "live_feedback": [],
        "processor_key": 0,   # increment to reset the webrtc processor
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ══════════════════════════════════════════════════════════════════════════════
# DEMO / SIMULATION ENGINE
# ══════════════════════════════════════════════════════════════════════════════
EXERCISE_ERRORS_MAP = {
    "BackSquat":     ["knees_inward", "knees_forward", "rounded_back", "shallow_squat"],
    "OverheadPress": ["elbow_error", "knees_error", "spine_error"],
    "BarbellRow":    ["lumbar_error", "torso_angle_error"],
}
FEEDBACK_MAP = {
    "knees_inward":      "Keep knees tracking over your toes.",
    "knees_forward":     "Don't let knees travel past toes.",
    "rounded_back":      "Brace core — neutral spine.",
    "shallow_squat":     "Squat deeper — aim for parallel.",
    "elbow_error":       "Keep elbows at 45° overhead.",
    "knees_error":       "Soft knees — avoid hyperextension.",
    "spine_error":       "Avoid excessive lumbar extension.",
    "lumbar_error":      "Flat back throughout the pull.",
    "torso_angle_error": "Hold torso angle — don't let hips rise.",
}


def sim_rep(exercise: str, n: int) -> dict:
    errs   = EXERCISE_ERRORS_MAP[exercise]
    base   = max(52, 94 - n * 0.5 + random.gauss(0, 4))
    errors = {e: random.random() < (0.12 + n * 0.018) for e in errs}
    score  = max(18, min(100, int(base - sum(errors.values()) * 11)))
    return {
        "rep_number": n,
        "score":      score,
        "errors":     errors,
        "feedback":   [FEEDBACK_MAP[e] for e, v in errors.items() if v],
        "timestamp":  datetime.datetime.now().isoformat(),
    }


def trend_calc(scores: list):
    if len(scores) < 4:
        return "insufficient data", 0.0
    s = float(np.polyfit(range(len(scores)), scores, 1)[0])
    return ("declining" if s < -3 else "improving" if s > 2 else "stable"), s


def err_freq(reps: list) -> dict:
    f = {}
    for r in reps:
        for e, v in r["errors"].items():
            if v:
                f[e] = f.get(e, 0) + 1
    return dict(sorted(f.items(), key=lambda x: -x[1]))


def overload_ok(reps: list) -> bool:
    return len(reps) >= 3 and all(r["score"] >= 80 for r in reps[-3:])


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="padding:12px 0 20px;">
      <div style="display:flex;align-items:baseline;gap:6px;">
        <span style="font-family:'Bebas Neue',sans-serif;font-size:32px;letter-spacing:0.1em;
          background:linear-gradient(135deg,#a8ff3e,#00e6a0);
          -webkit-background-clip:text;-webkit-text-fill-color:transparent;
          background-clip:text;">FORMIQ</span>
        <span style="font-size:10px;font-weight:700;letter-spacing:0.16em;
          color:#3d7055;text-transform:uppercase;">AI Coach</span>
      </div>
    </div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown(section_label("Session Config"), unsafe_allow_html=True)

    ex = st.selectbox(
        "Exercise",
        ["BackSquat", "OverheadPress", "BarbellRow"],
        index=["BackSquat", "OverheadPress", "BarbellRow"].index(
            st.session_state.exercise),
    )
    # Reset processor when exercise changes
    if ex != st.session_state.exercise:
        st.session_state.exercise = ex
        st.session_state.reps     = []
        st.session_state.processor_key += 1

    athlete = st.text_input("Athlete ID", value=st.session_state.athlete_id)
    st.session_state.athlete_id = athlete

    st.markdown(section_label("Camera"), unsafe_allow_html=True)
    st.selectbox("Source", ["Webcam 0", "Webcam 1", "IP Stream"])

    st.markdown("<br/>", unsafe_allow_html=True)
    st.divider()

    if st.button("↺  RESET SESSION"):
        st.session_state.reps          = []
        st.session_state.live_score    = 0
        st.session_state.live_errors   = {}
        st.session_state.live_feedback = []
        st.session_state.session_start = None
        st.session_state.processor_key += 1
        st.rerun()

    st.markdown("<br/>", unsafe_allow_html=True)
    if st.button("＋  SIMULATE REP"):
        n   = len(st.session_state.reps) + 1
        rec = sim_rep(st.session_state.exercise, n)
        st.session_state.reps.append(rec)
        st.session_state.live_score    = rec["score"]
        st.session_state.live_errors   = rec["errors"]
        st.session_state.live_feedback = rec["feedback"]
        if not st.session_state.session_start:
            st.session_state.session_start = datetime.datetime.now()
        st.rerun()

    st.divider()
    st.markdown(section_label("Export"), unsafe_allow_html=True)
    if st.session_state.reps:
        st.download_button(
            "⬇  DOWNLOAD JSON",
            data=json.dumps({
                "exercise":   st.session_state.exercise,
                "athlete_id": athlete,
                "reps":       st.session_state.reps,
            }, indent=2),
            file_name=f"formiq_{ex}_{datetime.date.today()}.json",
            mime="application/json",
            use_container_width=True,
        )

    # Load session log from main.py
    st.divider()
    st.markdown(section_label("Load Session Log"), unsafe_allow_html=True)
    log_dir = Path("logs")
    if log_dir.exists():
        logs = sorted(log_dir.glob("*.json"), reverse=True)
        if logs:
            sel = st.selectbox("Session", [f.name for f in logs])
            if st.button("📂  LOAD LOG"):
                with open(log_dir / sel) as f:
                    data = json.load(f)
                st.session_state.reps       = data.get("rep_log", [])
                st.session_state.exercise   = data.get("exercise", "BackSquat")
                st.session_state.athlete_id = data.get("athlete_id", "Athlete")
                st.rerun()

    st.markdown("<br/>", unsafe_allow_html=True)
    is_live = WEBRTC_AVAILABLE
    st.markdown(status_dot(is_live), unsafe_allow_html=True)
    if not WEBRTC_AVAILABLE:
        st.markdown(
            "<p style='font-size:10px;color:#3d7055;margin-top:6px;'>"
            "pip install streamlit-webrtc aiortc</p>",
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div style="margin-bottom:24px;">
  <h1 style="font-size:clamp(28px,4.5vw,48px);margin:0;line-height:1;
    background:linear-gradient(135deg,#a8ff3e 0%,#00e6a0 45%,#00c8e0 100%);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">
    EXERCISE FORM ASSESSMENT
  </h1>
  <p style="color:#3d7055;font-size:12px;margin-top:5px;letter-spacing:0.06em;
    font-family:'Space Grotesk',sans-serif;">
    {st.session_state.exercise}&ensp;·&ensp;{athlete}&ensp;·&ensp;{datetime.date.today().strftime('%d %b %Y')}
  </p>
</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# CAMERA + SCORE ROW
# ══════════════════════════════════════════════════════════════════════════════
cam_col, score_col = st.columns([1.6, 1], gap="large")

with cam_col:
    if WEBRTC_AVAILABLE:
        # ── Capture exercise NOW in main thread before lambda runs ────────────
        # The lambda runs in a background thread where st.session_state
        # is not accessible — must capture value first.
        _exercise = st.session_state.get("exercise", "BackSquat")
        _pkey     = st.session_state.get("processor_key", 0)

        def _make_processor():
            try:
                return FormVideoProcessor(_exercise)
            except Exception as e:
                import sys
                print(f"[FormIQ] Processor init error: {e}", file=sys.stderr)
                return FormVideoProcessor("BackSquat")

        # Multiple STUN servers for better connectivity
        RTC_CONFIG = RTCConfiguration({
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": ["stun:stun1.l.google.com:19302"]},
                {"urls": ["stun:stun2.l.google.com:19302"]},
            ]
        })

        st.markdown('<div class="webrtc-wrap">', unsafe_allow_html=True)
        try:
            ctx = webrtc_streamer(
                key=f"formiq-{_exercise}-{_pkey}",
                video_processor_factory=_make_processor,
                rtc_configuration=RTC_CONFIG,
                media_stream_constraints={
                    "video": {
                        "width":       {"ideal": 1280, "min": 640},
                        "height":      {"ideal": 720,  "min": 480},
                        "frameRate":   {"ideal": 30,   "max": 30},
                        "facingMode":  "user",
                    },
                    "audio": False,
                },
                async_processing=True,
            )
        except Exception as e:
            ctx = None
            st.markdown(
                '<p style="color:#ff4d6d;font-size:12px;text-align:center;padding:8px;">' +
                'Camera connection failed — check browser permissions and try again.</p>',
                unsafe_allow_html=True,
            )
            import sys
            print(f"[FormIQ] webrtc_streamer error: {e}", file=sys.stderr)
        st.markdown('</div>', unsafe_allow_html=True)

        # Sync new reps from processor to session state (main thread only)
        if ctx is not None and ctx.video_processor:
            try:
                new_reps = ctx.video_processor.flush_new_reps()
                if new_reps:
                    st.session_state.reps.extend(new_reps)
                    last = new_reps[-1]
                    st.session_state.live_score    = last["score"]
                    st.session_state.live_errors   = last["errors"]
                    st.session_state.live_feedback = last["feedback"]
                    if not st.session_state.session_start:
                        st.session_state.session_start = datetime.datetime.now()
            except Exception:
                pass

        st.markdown(
            "<p style='font-size:10px;color:#3d7055;text-align:center;"
            "margin-top:4px;font-family:Space Grotesk;letter-spacing:0.05em;'>"
            "MediaPipe · 33 keypoints · Real-time skeleton + HUD</p>",
            unsafe_allow_html=True,
        )
    else:
        # ── File-based live feed from main.py ─────────────────────────────────
        # main.py writes frames to .formiq_live/frame.jpg at ~12fps.
        # We read and display here with st.empty() for smooth in-place updates.
        import base64 as _b64, time as _t

        LIVE_DIR   = Path(".formiq_live")
        FRAME_FILE = LIVE_DIR / "frame.jpg"
        STATE_FILE = LIVE_DIR / "state.json"

        live_frame_b64 = None
        live_state     = None

        try:
            if FRAME_FILE.exists() and (_t.time() - FRAME_FILE.stat().st_mtime) < 3:
                live_frame_b64 = _b64.b64encode(FRAME_FILE.read_bytes()).decode()
            if STATE_FILE.exists() and (_t.time() - STATE_FILE.stat().st_mtime) < 3:
                live_state = json.loads(STATE_FILE.read_text())
        except Exception:
            pass

        # Sync live state into session state when main.py is running
        if live_state:
            try:
                st.session_state.live_score  = live_state.get("score", 0)
                st.session_state.live_errors = live_state.get("errors", {})
                rep_count = live_state.get("rep_count", 0)
                if rep_count > len(st.session_state.reps):
                    st.session_state.reps.append({
                        "rep_number": rep_count,
                        "score":      live_state.get("score", 0),
                        "errors":     live_state.get("errors", {}),
                        "feedback":   live_state.get("feedback", []),
                        "timestamp":  datetime.datetime.now().isoformat(),
                    })
                    if not st.session_state.session_start:
                        st.session_state.session_start = datetime.datetime.now()
            except Exception:
                pass

        if live_frame_b64:
            phase   = live_state.get("phase", "") if live_state else ""
            rep_cnt = live_state.get("rep_count", 0) if live_state else 0
            score_v = live_state.get("score", 0) if live_state else 0
            sc_col  = "#a8ff3e" if score_v >= 75 else "#ffb800" if score_v >= 50 else "#ff4d6d"
            st.markdown(
                f'<div style="position:relative;border-radius:16px;overflow:hidden;'
                f'border:2px solid rgba(0,230,160,0.4);'
                f'box-shadow:0 0 32px rgba(0,230,160,0.12);">'
                f'<img src="data:image/jpeg;base64,{live_frame_b64}" '
                f'style="width:100%;display:block;border-radius:14px;" loading="eager"/>'
                f'<div style="position:absolute;top:10px;right:10px;'
                f'background:rgba(10,20,12,0.8);border:1px solid rgba(255,77,109,0.5);'
                f'color:#ff4d6d;font-family:Space Grotesk,sans-serif;font-size:10px;'
                f'font-weight:700;padding:3px 10px;border-radius:99px;letter-spacing:0.1em;">'
                f'● LIVE</div>'
                f'<div style="position:absolute;bottom:10px;left:10px;display:flex;gap:8px;">'
                f'<span style="background:rgba(10,20,12,0.8);border:1px solid rgba(0,230,160,0.3);'
                f'color:#00e6a0;font-family:Space Grotesk;font-size:10px;font-weight:600;'
                f'padding:3px 10px;border-radius:99px;">{rep_cnt} REPS</span>'
                f'<span style="background:rgba(10,20,12,0.8);border:1px solid {sc_col}55;'
                f'color:{sc_col};font-family:Space Grotesk;font-size:10px;font-weight:600;'
                f'padding:3px 10px;border-radius:99px;">{score_v}/100</span>'
                f'</div></div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                "<p style='font-size:10px;color:#3d7055;text-align:center;margin-top:4px;"
                "font-family:Space Grotesk;letter-spacing:0.05em;'>Live feed from main.py"
                " · MediaPipe skeleton overlay active</p>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="aspect-ratio:16/9;background:rgba(2,10,14,0.9);'
                'border-radius:16px;border:1px solid rgba(0,230,160,0.1);'
                'display:flex;flex-direction:column;align-items:center;'
                'justify-content:center;gap:12px;">'
                '<div style="font-size:32px;opacity:0.15;">📷</div>'
                '<p style="font-family:Bebas Neue,sans-serif;font-size:16px;'
                'letter-spacing:0.12em;color:#3d7055;margin:0;">CAMERA OFFLINE</p>'
                '<code style="background:rgba(0,230,160,0.08);color:#00e6a0;'
                'padding:4px 12px;border-radius:8px;font-size:11px;">'
                'python main.py --exercise BackSquat</code>'
                '<p style="font-size:10px;color:#3d7055;text-align:center;'
                'max-width:280px;">Run main.py in a second terminal to start the live feed</p>'
                '</div>',
                unsafe_allow_html=True,
            )

with score_col:
    # Score ring card
    score     = st.session_state.live_score
    ring_html = score_ring_svg(score, 120)
    sl        = section_label("Last Rep Score")
    if st.session_state.live_errors:
        pills = " ".join([pill(e, v) for e, v in st.session_state.live_errors.items()])
    else:
        pills = '<span style="color:#3d7055;font-size:12px;">No rep data yet</span>'

    st.markdown(
        card(
            sl
            + '<div style="display:flex;justify-content:center;margin:8px 0 14px;">'
            + ring_html + '</div>'
            + '<div style="display:flex;flex-wrap:wrap;gap:6px;">' + pills + '</div>',
            "margin-bottom:12px;",
        ),
        unsafe_allow_html=True,
    )

    # Feedback card
    if st.session_state.live_feedback:
        fb_items = "".join([
            '<div style="display:flex;gap:8px;padding:6px 0;'
            'border-bottom:1px solid rgba(0,230,160,0.07);">'
            '<span style="color:#ffb800;font-size:12px;flex-shrink:0;">→</span>'
            f'<span style="font-size:12px;color:rgba(232,245,238,0.8);line-height:1.5;">{fb}</span>'
            '</div>'
            for fb in st.session_state.live_feedback
        ])
        st.markdown(
            card(section_label("Corrective Feedback") + fb_items),
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<br/>", unsafe_allow_html=True)
t1, t2, t3, t4 = st.tabs(["📊  Session", "🔁  Rep Log", "📈  Analytics", "📖  Setup"])


# ─────────────────────────────────────────────────────────
# TAB 1 — Session
# ─────────────────────────────────────────────────────────
with t1:
    reps   = st.session_state.reps
    scores = [r["score"] for r in reps]

    k1, k2, k3, k4 = st.columns(4)
    with k1: st.metric("Total Reps", len(reps))
    with k2: st.metric("Avg Score",  f"{np.mean(scores):.0f}" if scores else "—")
    with k3: st.metric("Best Rep",   max(scores) if scores else "—")
    with k4:
        dur = "—"
        if st.session_state.session_start:
            s   = int((datetime.datetime.now() - st.session_state.session_start).total_seconds())
            dur = f"{s//60}:{s%60:02d}"
        st.metric("Duration", dur)

    st.markdown("<br/>", unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="large")

    with c1:
        tl, slope = trend_calc(scores)
        ov        = overload_ok(reps)
        freq      = err_freq(reps)
        top_err   = list(freq.keys())[0] if freq else None

        slope_span   = (f'<span style="font-size:11px;color:#3d7055;margin-left:8px;">'
                        f'{slope:+.1f} pts/rep</span>') if len(scores) >= 4 else ""
        overload_span = ('<span style="color:#a8ff3e;font-weight:700;font-size:13px;">'
                         '✓ Ready to increase load</span>'
                         if ov else
                         '<span style="color:#3d7055;font-size:12px;">Need 3 reps ≥80</span>')
        top_span      = (top_err.replace("_", " ").title()
                         if top_err else "None — clean session")

        html = (
            section_label("Session Status")
            + '<div style="display:flex;flex-direction:column;gap:16px;">'
            + '<div><p style="font-size:11px;color:#3d7055;margin-bottom:6px;">FORM TREND</p>'
            + trend_chip(tl) + slope_span + '</div>'
            + '<div><p style="font-size:11px;color:#3d7055;margin-bottom:5px;">PROGRESSIVE OVERLOAD</p>'
            + overload_span + '</div>'
            + '<div><p style="font-size:11px;color:#3d7055;margin-bottom:5px;">TOP ERROR</p>'
            + f'<span style="font-size:13px;color:#ff8fa3;font-weight:600;">{top_span}</span></div>'
            + '</div>'
        )
        st.markdown(card(html), unsafe_allow_html=True)

    with c2:
        fatigue = len(scores) >= 4 and tl == "declining"
        plateau = len(scores) >= 5 and (max(scores[-5:]) - min(scores[-5:])) <= 5
        alerts  = []
        if fatigue: alerts.append(("⚡", "FATIGUE", "Form declining — rest or reduce load.",
                                   "#ff4d6d", "rgba(255,77,109,0.1)"))
        if ov:      alerts.append(("🚀", "OVERLOAD READY", "3 clean reps ≥80 — add weight.",
                                   "#a8ff3e", "rgba(168,255,62,0.08)"))
        if plateau: alerts.append(("📊", "PLATEAU", "Scores stable — vary load or cues.",
                                   "#ffb800", "rgba(255,184,0,0.08)"))
        if not alerts:
            alerts.append(("✓", "ALL CLEAR", "No alerts — keep training.",
                           "#00e6a0", "rgba(0,230,160,0.06)"))

        alerts_html = "".join([alert_box(i, t, m, c, b) for i, t, m, c, b in alerts])
        st.markdown(card(section_label("Alerts") + alerts_html), unsafe_allow_html=True)

    if scores:
        st.markdown("<br/>", unsafe_allow_html=True)
        bars = ""
        for r in reps:
            s = r["score"]
            c = "#a8ff3e" if s >= 75 else "#ffb800" if s >= 50 else "#ff4d6d"
            bars += (
                '<div style="display:flex;flex-direction:column;align-items:center;'
                'gap:3px;flex:1;min-width:18px;">'
                f'<span style="font-size:9px;color:{c};font-weight:700;">{s}</span>'
                '<div style="width:100%;background:rgba(0,80,40,0.35);border-radius:5px;'
                'height:68px;display:flex;align-items:flex-end;overflow:hidden;">'
                f'<div style="width:100%;height:{s}%;background:{c};'
                f'box-shadow:0 0 8px {c}66;border-radius:4px;"></div></div>'
                f'<span style="font-size:8px;color:#3d7055;">R{r["rep_number"]}</span>'
                '</div>'
            )
        st.markdown(
            card(section_label("Rep Scores")
                 + '<div style="display:flex;align-items:flex-end;gap:4px;'
                 'height:90px;overflow-x:auto;">' + bars + '</div>'),
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────
# TAB 2 — Rep Log
# ─────────────────────────────────────────────────────────
with t2:
    reps = st.session_state.reps
    if not reps:
        st.markdown(card(
            '<div style="text-align:center;padding:32px 0;">'
            '<div style="font-size:32px;opacity:0.2;margin-bottom:10px;">🔁</div>'
            '<p style="font-family:Bebas Neue,sans-serif;font-size:16px;'
            'letter-spacing:0.08em;color:#3d7055;">No reps yet</p>'
            '</div>'
        ), unsafe_allow_html=True)
    else:
        for rep in reversed(reps):
            s   = rep["score"]
            det = [e for e, v in rep["errors"].items() if v]
            ph  = (" ".join([pill(e, True) for e in det])
                   if det else
                   '<span style="color:#00e6a0;font-size:11px;">✓ Clean rep</span>')
            with st.expander(
                f"Rep {rep['rep_number']}  —  {s}/100",
                expanded=(rep["rep_number"] == len(reps)),
            ):
                lc, rc = st.columns([1, 2])
                with lc:
                    ring = score_ring_svg(s, 96)
                    st.markdown(
                        '<div style="display:flex;justify-content:center;margin:8px 0;">'
                        + ring + '</div>',
                        unsafe_allow_html=True,
                    )
                with rc:
                    fb_html = "".join([
                        f'<p style="font-size:12px;color:#ff8fa3;margin:3px 0;">→ {fb}</p>'
                        for fb in rep["feedback"]
                    ]) or '<p style="font-size:12px;color:#00e6a0;">Clean rep.</p>'
                    st.markdown(
                        f'<div style="padding-top:6px;">'
                        f'<div style="display:flex;flex-wrap:wrap;gap:5px;margin-bottom:8px;">{ph}</div>'
                        + fb_html
                        + f'<p style="font-size:10px;color:#3d7055;margin-top:8px;">'
                        f'{rep.get("timestamp","")[:19]}</p></div>',
                        unsafe_allow_html=True,
                    )


# ─────────────────────────────────────────────────────────
# TAB 3 — Analytics
# ─────────────────────────────────────────────────────────
with t3:
    reps   = st.session_state.reps
    freq   = err_freq(reps)
    scores = [r["score"] for r in reps]

    if not reps:
        st.info("Log some reps to see analytics.")
    else:
        la, ra = st.columns(2, gap="large")

        with la:
            mf = max(freq.values()) if freq else 1
            fi = ""
            for err, cnt in freq.items():
                pct = int(cnt / mf * 100)
                col = "#ff4d6d" if pct > 50 else "#ffb800" if pct > 25 else "#00e6a0"
                fi += (
                    '<div style="margin-bottom:12px;">'
                    '<div style="display:flex;justify-content:space-between;margin-bottom:4px;">'
                    f'<span style="font-size:12px;color:rgba(232,245,238,0.85);">'
                    f'{err.replace("_"," ").title()}</span>'
                    f'<span style="font-size:11px;color:{col};font-weight:700;">'
                    f'{cnt}/{len(reps)}</span></div>'
                    + mini_bar(pct, col) + '</div>'
                )
            empty = '<p style="color:#3d7055;font-size:12px;">No errors detected.</p>'
            st.markdown(
                card(section_label("Error Frequency") + (fi or empty)),
                unsafe_allow_html=True,
            )

        with ra:
            bk = {"≥80 — Good": 0, "60–79 — OK": 0, "<60 — Poor": 0}
            bc = {"≥80 — Good": "#a8ff3e", "60–79 — OK": "#ffb800", "<60 — Poor": "#ff4d6d"}
            for s in scores:
                if s >= 80:   bk["≥80 — Good"] += 1
                elif s >= 60: bk["60–79 — OK"] += 1
                else:         bk["<60 — Poor"] += 1
            di = ""
            for lb, cnt in bk.items():
                pct = int(cnt / len(scores) * 100) if scores else 0
                di += (
                    '<div style="margin-bottom:12px;">'
                    '<div style="display:flex;justify-content:space-between;margin-bottom:4px;">'
                    f'<span style="font-size:12px;color:rgba(232,245,238,0.85);">{lb}</span>'
                    f'<span style="font-size:11px;color:{bc[lb]};font-weight:700;">'
                    f'{cnt} ({pct}%)</span></div>'
                    + mini_bar(pct, bc[lb]) + '</div>'
                )
            st.markdown(
                card(section_label("Score Distribution") + di),
                unsafe_allow_html=True,
            )

        if freq:
            top   = list(freq.keys())[0]
            drills = {
                "knees_inward":      "3×12 banded squats — cue knee tracking outward.",
                "knees_forward":     "Wall squat — toes 2cm from wall.",
                "rounded_back":      "Paused RDL — reinforce neutral spine.",
                "shallow_squat":     "Box squat to low box — build depth.",
                "elbow_error":       "Z-press (seated) — isolate vertical path.",
                "knees_error":       "Tempo OHP 3-1-1 — slow eccentric.",
                "spine_error":       "Dead-stop OHP from rack pins.",
                "lumbar_error":      "Chest-supported row — remove spinal load.",
                "torso_angle_error": "Paused rows — hold torso at contraction.",
            }
            tip = drills.get(top, "Review form with a qualified coach.")
            st.markdown("<br/>", unsafe_allow_html=True)
            st.markdown(card(
                section_label("Top Coaching Recommendation")
                + '<div style="display:flex;gap:14px;align-items:flex-start;">'
                '<span style="font-size:26px;flex-shrink:0;">🎯</span>'
                '<div>'
                f'<p style="font-family:Bebas Neue,sans-serif;font-size:16px;'
                f'letter-spacing:0.06em;color:#ff8fa3;margin-bottom:5px;">'
                f'{top.replace("_"," ").upper()}</p>'
                f'<p style="font-size:13px;color:rgba(232,245,238,0.82);line-height:1.6;">{tip}</p>'
                '</div></div>'
            ), unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# TAB 4 — Setup
# ─────────────────────────────────────────────────────────
with t4:
    steps = [
        ("01", "Install Dependencies",
         "pip install mediapipe opencv-python numpy scipy\n"
         "pip install scikit-learn streamlit joblib\n"
         "pip install streamlit-webrtc aiortc av",
         "Python 3.10+ · CPU-only · GPU optional"),
        ("02", "Train Models",
         "python train.py --exercise Squat\n"
         "python train.py --exercise OHP\n"
         "python train.py --exercise BarbellRow\n"
         "# Or all at once:\npython train.py --exercise all",
         "Models saved to models/ · 30–120 min per exercise on CPU"),
        ("03", "Launch Dashboard (with live camera)",
         "streamlit run app.py",
         "Browser opens at http://localhost:8501 · Click START in the camera panel"),
        ("04", "CLI Live Mode (OpenCV window)",
         "python main.py --exercise BackSquat\n"
         "python main.py --exercise OverheadPress\n"
         "python main.py --exercise BarbellRow\n"
         "# Press Q to quit — log saves to logs/",
         "Auto-resolves model from models/ · session JSON saved to logs/"),
        ("05", "Load Session in Dashboard",
         "# After running main.py, load the log in the sidebar:\n"
         "# Sidebar → Load Session Log → select file → LOAD LOG",
         "Full rep-by-rep review with analytics"),
    ]
    for num, title, code, note in steps:
        st.markdown(card(
            '<div style="display:flex;gap:14px;align-items:flex-start;">'
            '<div style="flex-shrink:0;width:32px;height:32px;border-radius:8px;'
            'background:linear-gradient(135deg,rgba(168,255,62,0.15),rgba(0,230,160,0.1));'
            'border:1px solid rgba(0,230,160,0.25);display:flex;align-items:center;'
            'justify-content:center;font-family:Bebas Neue,sans-serif;'
            f'font-size:13px;letter-spacing:0.06em;color:#00e6a0;">{num}</div>'
            '<div style="flex:1;min-width:0;">'
            f'<p style="font-family:Bebas Neue,sans-serif;font-size:15px;'
            f'letter-spacing:0.06em;color:#e8f5ee;margin-bottom:7px;">{title}</p>'
            f'<pre style="background:rgba(0,10,8,0.7);border:1px solid rgba(0,230,160,0.1);'
            f'border-radius:8px;padding:10px;font-size:11px;color:#7bbf9a;overflow-x:auto;'
            f'margin-bottom:6px;font-family:JetBrains Mono,monospace;line-height:1.6;">{code}</pre>'
            f'<p style="font-size:11px;color:#3d7055;">{note}</p>'
            '</div></div>',
            "margin-bottom:10px;",
        ), unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# AUTO-REFRESH  — poll file-based feed at 6fps (every 160ms)
# Only rerun when main.py is actually writing fresh frames.
# ══════════════════════════════════════════════════════════════════════════════
try:
    import time as _refresh_time
    _ff = Path(".formiq_live") / "frame.jpg"
    _is_live = _ff.exists() and (_refresh_time.time() - _ff.stat().st_mtime) < 3
    if _is_live:
        time.sleep(0.16)   # ~6fps refresh — smooth without hammering the CPU
        st.rerun()
    elif WEBRTC_AVAILABLE and "ctx" in dir() and ctx is not None:
        try:
            if ctx.state.playing:
                time.sleep(0.16)
                st.rerun()
        except Exception:
            pass
except Exception:
    pass