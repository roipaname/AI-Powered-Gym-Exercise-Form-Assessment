"""
app.py — Glassmorphic Streamlit Frontend for Gym AQ System
Run: streamlit run app.py
"""

import streamlit as st
import numpy as np
import json
import time
import random
import datetime
from pathlib import Path

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="FormIQ — AI Gym Coach",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inject custom CSS (glassmorphic green theme) ──────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Root palette ── */
:root {
  --g50:  #f0fdf4;
  --g100: #dcfce7;
  --g200: #bbf7d0;
  --g300: #86efac;
  --g400: #4ade80;
  --g500: #22c55e;
  --g600: #16a34a;
  --g700: #15803d;
  --g800: #166534;
  --g900: #14532d;
  --dark: #050e08;
  --panel: rgba(10, 28, 16, 0.55);
  --glass: rgba(20, 83, 45, 0.22);
  --glass-border: rgba(74, 222, 128, 0.18);
  --glow: rgba(34, 197, 94, 0.25);
  --text-primary: #ecfdf5;
  --text-muted: #86efac;
  --text-dim: #4ade80;
  --accent: #4ade80;
  --danger: #f87171;
  --warn: #fbbf24;
}

/* ── Global reset ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; }

html, body, [data-testid="stAppViewContainer"],
[data-testid="stApp"], .main, .block-container {
  background: transparent !important;
}

/* Deep green gradient background */
[data-testid="stApp"]::before {
  content: '';
  position: fixed;
  inset: 0;
  background:
    radial-gradient(ellipse 80% 50% at 20% 10%, rgba(22,163,74,0.18) 0%, transparent 60%),
    radial-gradient(ellipse 60% 60% at 80% 80%, rgba(21,128,61,0.14) 0%, transparent 60%),
    radial-gradient(ellipse 100% 100% at 50% 50%, #050e08 0%, #040c06 100%);
  z-index: -1;
}

/* Subtle grid texture */
[data-testid="stApp"]::after {
  content: '';
  position: fixed;
  inset: 0;
  background-image:
    linear-gradient(rgba(74,222,128,0.03) 1px, transparent 1px),
    linear-gradient(90deg, rgba(74,222,128,0.03) 1px, transparent 1px);
  background-size: 48px 48px;
  z-index: -1;
  pointer-events: none;
}

/* ── Typography ── */
body, .stMarkdown, p, span, label, div {
  font-family: 'DM Sans', sans-serif !important;
  color: var(--text-primary) !important;
}
h1, h2, h3, h4 {
  font-family: 'Syne', sans-serif !important;
  color: var(--text-primary) !important;
  font-weight: 700 !important;
  letter-spacing: -0.02em;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: var(--panel) !important;
  border-right: 1px solid var(--glass-border) !important;
  backdrop-filter: blur(24px) !important;
}
[data-testid="stSidebar"] * {
  color: var(--text-primary) !important;
  font-family: 'DM Sans', sans-serif !important;
}

/* ── Selectbox / inputs ── */
.stSelectbox > div > div,
.stSlider > div, .stNumberInput > div > div {
  background: var(--glass) !important;
  border: 1px solid var(--glass-border) !important;
  border-radius: 12px !important;
  color: var(--text-primary) !important;
}
.stSelectbox [data-baseweb="select"] > div {
  background: transparent !important;
  color: var(--text-primary) !important;
  border: none !important;
}

/* ── Buttons ── */
.stButton > button {
  background: linear-gradient(135deg, rgba(34,197,94,0.22), rgba(21,128,61,0.18)) !important;
  border: 1px solid var(--glass-border) !important;
  border-radius: 12px !important;
  color: var(--g300) !important;
  font-family: 'Syne', sans-serif !important;
  font-weight: 600 !important;
  font-size: 14px !important;
  padding: 10px 24px !important;
  letter-spacing: 0.04em !important;
  transition: all 0.2s ease !important;
  width: 100% !important;
  backdrop-filter: blur(8px) !important;
}
.stButton > button:hover {
  background: linear-gradient(135deg, rgba(74,222,128,0.28), rgba(34,197,94,0.22)) !important;
  border-color: rgba(74,222,128,0.45) !important;
  box-shadow: 0 0 24px var(--glow) !important;
  color: #fff !important;
  transform: translateY(-1px);
}

/* ── Metric cards ── */
[data-testid="metric-container"] {
  background: var(--glass) !important;
  border: 1px solid var(--glass-border) !important;
  border-radius: 16px !important;
  padding: 16px !important;
  backdrop-filter: blur(16px) !important;
}
[data-testid="stMetricValue"] {
  font-family: 'Syne', sans-serif !important;
  font-size: 2rem !important;
  font-weight: 800 !important;
  color: var(--g300) !important;
}
[data-testid="stMetricLabel"] {
  font-size: 12px !important;
  text-transform: uppercase !important;
  letter-spacing: 0.1em !important;
  color: var(--text-muted) !important;
}
[data-testid="stMetricDelta"] svg { display: none; }

/* ── Progress bars ── */
.stProgress > div > div > div {
  background: linear-gradient(90deg, var(--g600), var(--g400)) !important;
  border-radius: 99px !important;
  box-shadow: 0 0 12px var(--glow) !important;
}
.stProgress > div > div {
  background: rgba(20, 83, 45, 0.4) !important;
  border-radius: 99px !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
  background: var(--glass) !important;
  border-radius: 12px !important;
  border: 1px solid var(--glass-border) !important;
  padding: 4px !important;
  gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
  background: transparent !important;
  color: var(--text-muted) !important;
  font-family: 'Syne', sans-serif !important;
  font-size: 13px !important;
  font-weight: 600 !important;
  border-radius: 8px !important;
  padding: 8px 20px !important;
  border: none !important;
  letter-spacing: 0.05em !important;
}
.stTabs [aria-selected="true"] {
  background: linear-gradient(135deg, rgba(34,197,94,0.25), rgba(21,128,61,0.2)) !important;
  color: var(--g300) !important;
  box-shadow: 0 0 16px rgba(34,197,94,0.2) !important;
}
.stTabs [data-baseweb="tab-border"] { display: none !important; }

/* ── Expanders ── */
.streamlit-expanderHeader {
  background: var(--glass) !important;
  border: 1px solid var(--glass-border) !important;
  border-radius: 12px !important;
  font-family: 'Syne', sans-serif !important;
  color: var(--text-primary) !important;
}

/* ── Divider ── */
hr {
  border-color: var(--glass-border) !important;
  margin: 16px 0 !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header, [data-testid="stToolbar"] { visibility: hidden; }
.block-container { padding-top: 2rem !important; padding-bottom: 2rem !important; }

/* ── Toast / info boxes ── */
.stAlert {
  background: var(--glass) !important;
  border: 1px solid var(--glass-border) !important;
  border-radius: 12px !important;
  backdrop-filter: blur(12px) !important;
}

/* ── Custom scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--g700); border-radius: 99px; }
::-webkit-scrollbar-thumb:hover { background: var(--g500); }
</style>
""", unsafe_allow_html=True)

# ── Utility HTML components ───────────────────────────────────────────────────

def glass_card(content_html: str, extra_style: str = "") -> str:
    return f"""
    <div style="
      background: rgba(20,83,45,0.22);
      border: 1px solid rgba(74,222,128,0.18);
      border-radius: 20px;
      padding: 24px;
      backdrop-filter: blur(20px);
      -webkit-backdrop-filter: blur(20px);
      {extra_style}
    ">{content_html}</div>
    """

def score_ring(score: int, size: int = 120) -> str:
    """SVG donut ring for score display."""
    pct   = max(0, min(100, score))
    circ  = 2 * 3.14159 * 42
    dash  = circ * pct / 100
    color = "#4ade80" if pct >= 75 else "#fbbf24" if pct >= 50 else "#f87171"
    glow  = "rgba(74,222,128,0.4)" if pct >= 75 else "rgba(251,191,36,0.4)" if pct >= 50 else "rgba(248,113,113,0.4)"
    return f"""
    <div style="display:flex;flex-direction:column;align-items:center;gap:8px;">
      <svg width="{size}" height="{size}" viewBox="0 0 100 100">
        <circle cx="50" cy="50" r="42" fill="none" stroke="rgba(20,83,45,0.6)" stroke-width="8"/>
        <circle cx="50" cy="50" r="42" fill="none" stroke="{color}" stroke-width="8"
          stroke-dasharray="{dash:.1f} {circ:.1f}" stroke-dashoffset="{circ/4:.1f}"
          stroke-linecap="round" style="filter:drop-shadow(0 0 6px {glow});"/>
        <text x="50" y="46" text-anchor="middle" font-size="22" font-weight="800"
          font-family="Syne,sans-serif" fill="{color}">{score}</text>
        <text x="50" y="60" text-anchor="middle" font-size="9" font-family="DM Sans,sans-serif"
          fill="rgba(134,239,172,0.7)">/100</text>
      </svg>
    </div>"""

def pill(label: str, detected: bool) -> str:
    bg   = "rgba(248,113,113,0.15)" if detected else "rgba(34,197,94,0.12)"
    bd   = "rgba(248,113,113,0.45)" if detected else "rgba(74,222,128,0.3)"
    col  = "#fca5a5" if detected else "#86efac"
    icon = "✕" if detected else "✓"
    return f"""<span style="
      background:{bg};border:1px solid {bd};border-radius:99px;
      color:{col};font-size:12px;font-family:DM Sans,sans-serif;font-weight:500;
      padding:4px 12px;display:inline-flex;align-items:center;gap:6px;white-space:nowrap;
    ">{icon} {label.replace('_',' ')}</span>"""

def trend_badge(label: str) -> str:
    cfg = {
        "improving":          ("↗", "#4ade80", "rgba(34,197,94,0.15)", "rgba(74,222,128,0.3)"),
        "stable":             ("→", "#a3e635", "rgba(163,230,53,0.12)", "rgba(163,230,53,0.25)"),
        "declining":          ("↘", "#f87171", "rgba(248,113,113,0.15)", "rgba(248,113,113,0.35)"),
        "insufficient data":  ("⋯", "#9ca3af", "rgba(156,163,175,0.12)", "rgba(156,163,175,0.25)"),
    }
    icon, col, bg, bd = cfg.get(label, cfg["insufficient data"])
    return f"""<span style="
      background:{bg};border:1px solid {bd};color:{col};
      font-family:Syne,sans-serif;font-weight:600;font-size:13px;
      padding:5px 14px;border-radius:99px;display:inline-flex;align-items:center;gap:6px;
    ">{icon} {label.upper()}</span>"""

def mini_bar(value: float, max_val: float, color: str = "#4ade80") -> str:
    pct = min(100, int(value / max_val * 100)) if max_val else 0
    return f"""
    <div style="width:100%;height:6px;background:rgba(20,83,45,0.5);border-radius:99px;overflow:hidden;">
      <div style="width:{pct}%;height:100%;background:{color};border-radius:99px;
        box-shadow:0 0 8px {color}55;transition:width 0.4s ease;"></div>
    </div>"""

# ── Session state init ────────────────────────────────────────────────────────

def init_state():
    defaults = {
        "reps":       [],
        "running":    False,
        "exercise":   "BackSquat",
        "athlete_id": "Athlete",
        "session_start": None,
        "live_score": 0,
        "live_errors": {},
        "live_feedback": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ── Demo data generators ──────────────────────────────────────────────────────

EXERCISE_ERRORS_MAP = {
    "BackSquat":     ["knees_inward", "knees_forward", "rounded_back", "shallow_squat"],
    "OverheadPress": ["elbow_error", "knees_error", "spine_error"],
    "BarbellRow":    ["lumbar_error", "asymmetry", "depth_error"],
}

FEEDBACK_MAP = {
    "knees_inward":  "Keep knees tracking over your toes.",
    "knees_forward": "Don't let knees travel too far past toes.",
    "rounded_back":  "Brace core and maintain neutral spine.",
    "shallow_squat": "Squat deeper — aim for parallel.",
    "elbow_error":   "Keep elbows at 45° overhead.",
    "knees_error":   "Keep knees soft, avoid hyperextension.",
    "spine_error":   "Avoid excessive lumbar extension.",
    "lumbar_error":  "Maintain flat back throughout the pull.",
    "asymmetry":     "Pull evenly — check elbow symmetry.",
    "depth_error":   "Pull bar all the way to lower chest.",
}

def simulate_rep(exercise: str, rep_num: int, fatigue: bool = False) -> dict:
    """Simulate one rep for demo mode."""
    errors_list = EXERCISE_ERRORS_MAP[exercise]
    base_score  = max(55, 95 - rep_num * (3 if fatigue else 0.5) + random.gauss(0, 5))
    errors      = {e: random.random() < (0.15 + rep_num * 0.02) for e in errors_list}
    detected    = [e for e, v in errors.items() if v]
    penalty     = len(detected) * 12
    score       = max(20, min(100, int(base_score - penalty)))
    feedback    = [FEEDBACK_MAP[e] for e in detected]
    return {
        "rep_number": rep_num,
        "score":      score,
        "errors":     errors,
        "feedback":   feedback,
        "timestamp":  datetime.datetime.now().isoformat(),
    }

def compute_trend(scores):
    if len(scores) < 4:
        return "insufficient data", 0.0
    slope = float(np.polyfit(range(len(scores)), scores, 1)[0])
    if slope < -3:   return "declining",  slope
    elif slope > 2:  return "improving",  slope
    return "stable", slope

def error_frequency(reps):
    freq = {}
    for r in reps:
        for e, v in r["errors"].items():
            if v: freq[e] = freq.get(e, 0) + 1
    return dict(sorted(freq.items(), key=lambda x: -x[1]))

def overload_ready(reps):
    if len(reps) < 3: return False
    return all(r["score"] >= 80 for r in reps[-3:])

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="padding:8px 0 24px;">
      <div style="font-family:Syne,sans-serif;font-size:22px;font-weight:800;
        color:#4ade80;letter-spacing:-0.02em;display:flex;align-items:center;gap:10px;">
        <span style="font-size:28px;">🏋️</span> FormIQ
      </div>
      <div style="font-size:11px;color:#4ade80;opacity:0.7;letter-spacing:0.12em;
        text-transform:uppercase;margin-top:2px;padding-left:38px;">
        AI Gym Coach
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown("<p style='font-size:11px;letter-spacing:0.1em;text-transform:uppercase;color:#4ade80;opacity:0.7;'>Session Config</p>", unsafe_allow_html=True)

    exercise = st.selectbox("Exercise", ["BackSquat", "OverheadPress", "BarbellRow"],
                            index=["BackSquat","OverheadPress","BarbellRow"].index(st.session_state.exercise))
    st.session_state.exercise = exercise

    athlete_id = st.text_input("Athlete ID", value=st.session_state.athlete_id)
    st.session_state.athlete_id = athlete_id

    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:11px;letter-spacing:0.1em;text-transform:uppercase;color:#4ade80;opacity:0.7;'>Camera</p>", unsafe_allow_html=True)
    camera_id = st.selectbox("Camera source", ["Webcam 0", "Webcam 1", "IP Stream"], index=0)

    st.markdown("<br/>", unsafe_allow_html=True)
    st.divider()

    # Control buttons
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("▶  Start"):
            st.session_state.running = True
            st.session_state.session_start = datetime.datetime.now()
    with col_b:
        if st.button("■  Stop"):
            st.session_state.running = False

    if st.button("↺  Reset Session"):
        st.session_state.reps = []
        st.session_state.running = False
        st.session_state.live_score = 0
        st.session_state.live_errors = {}
        st.session_state.live_feedback = []
        st.session_state.session_start = None
        st.rerun()

    st.markdown("<br/>", unsafe_allow_html=True)
    if st.button("+ Simulate Rep (Demo)"):
        rep_num = len(st.session_state.reps) + 1
        rec = simulate_rep(st.session_state.exercise, rep_num)
        st.session_state.reps.append(rec)
        st.session_state.live_score   = rec["score"]
        st.session_state.live_errors  = rec["errors"]
        st.session_state.live_feedback = rec["feedback"]
        st.rerun()

    st.divider()
    # Load/save
    st.markdown("<p style='font-size:11px;letter-spacing:0.1em;text-transform:uppercase;color:#4ade80;opacity:0.7;'>Export</p>", unsafe_allow_html=True)
    if st.session_state.reps:
        summary = {
            "exercise":   st.session_state.exercise,
            "athlete_id": st.session_state.athlete_id,
            "reps":       st.session_state.reps,
        }
        st.download_button("⬇ Download JSON",
            data=json.dumps(summary, indent=2),
            file_name=f"session_{st.session_state.exercise}_{datetime.date.today()}.json",
            mime="application/json",
            use_container_width=True,
        )

    st.markdown("<br/>", unsafe_allow_html=True)
    # Status indicator
    status_col = "#4ade80" if st.session_state.running else "#374151"
    status_txt = "LIVE" if st.session_state.running else "IDLE"
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:8px;padding:10px;
      background:rgba(20,83,45,0.2);border:1px solid rgba(74,222,128,0.15);border-radius:10px;">
      <span style="width:8px;height:8px;border-radius:50%;background:{status_col};
        box-shadow:0 0 8px {status_col};{'animation:pulse 1.5s infinite;' if st.session_state.running else ''}"></span>
      <span style="font-family:Syne,sans-serif;font-size:11px;font-weight:700;
        color:{status_col};letter-spacing:0.12em;">{status_txt}</span>
    </div>
    <style>
    @keyframes pulse {{0%,100%{{opacity:1}}50%{{opacity:0.4}}}}
    </style>
    """, unsafe_allow_html=True)

# ── Main area ─────────────────────────────────────────────────────────────────

# Page header
st.markdown(f"""
<div style="margin-bottom:32px;">
  <h1 style="font-size:clamp(28px,4vw,42px);font-weight:800;margin:0;
    background:linear-gradient(135deg,#86efac,#4ade80,#22c55e);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
    background-clip:text;">
    Exercise Form Assessment
  </h1>
  <p style="color:#86efac;opacity:0.75;font-size:15px;margin-top:6px;font-family:DM Sans,sans-serif;">
    {st.session_state.exercise} · {st.session_state.athlete_id} ·
    {datetime.date.today().strftime('%d %b %Y')}
  </p>
</div>
""", unsafe_allow_html=True)

# ── Live Panel + Camera Feed Row ──────────────────────────────────────────────
top_left, top_right = st.columns([1.4, 1], gap="large")

with top_left:
    # Camera feed placeholder
    feed_content = f"""
    <div style="aspect-ratio:16/9;background:rgba(5,14,8,0.85);border-radius:16px;
      border:1px solid rgba(74,222,128,0.18);display:flex;flex-direction:column;
      align-items:center;justify-content:center;gap:12px;overflow:hidden;position:relative;">
      <div style="position:absolute;inset:0;
        background:radial-gradient(ellipse at 50% 50%,rgba(22,163,74,0.06) 0%,transparent 70%);"></div>
      <div style="font-size:48px;opacity:0.3;">📷</div>
      <p style="color:#4ade80;opacity:0.5;font-size:13px;letter-spacing:0.05em;
        text-transform:uppercase;font-family:Syne,sans-serif;font-weight:600;">
        {'● CAMERA ACTIVE' if st.session_state.running else 'Camera offline — use demo mode'}
      </p>
      <p style="color:#86efac;opacity:0.35;font-size:11px;font-family:DM Sans,sans-serif;">
        Run python main.py for live webcam feed
      </p>
      {'<div style="position:absolute;top:12px;right:12px;background:rgba(248,113,113,0.2);border:1px solid rgba(248,113,113,0.4);color:#f87171;font-family:Syne,sans-serif;font-size:10px;font-weight:700;padding:4px 10px;border-radius:99px;letter-spacing:0.1em;">● REC</div>' if st.session_state.running else ''}
    </div>
    """
    st.markdown(feed_content, unsafe_allow_html=True)

    # Skeleton overlay hint
    if st.session_state.running:
        st.markdown("<p style='font-size:11px;color:#4ade80;opacity:0.5;text-align:center;margin-top:8px;'>MediaPipe skeleton overlay active · 33 keypoints</p>", unsafe_allow_html=True)

with top_right:
    # Live score ring + current rep detail
    score = st.session_state.live_score
    st.markdown(f"""
    {glass_card(f'''
      <p style="font-family:Syne,sans-serif;font-size:11px;font-weight:700;
        letter-spacing:0.12em;text-transform:uppercase;color:#4ade80;opacity:0.7;margin-bottom:16px;">
        Last Rep Score
      </p>
      <div style="display:flex;justify-content:center;margin-bottom:20px;">
        {score_ring(score, 130)}
      </div>
      <div style="display:flex;flex-wrap:wrap;gap:8px;">
        {''.join([pill(e, v) for e, v in st.session_state.live_errors.items()]) if st.session_state.live_errors else
         '<span style="color:#4ade80;opacity:0.5;font-size:13px;">No rep data yet</span>'}
      </div>
    ''', "margin-bottom:16px;")}
    """, unsafe_allow_html=True)

    # Feedback
    if st.session_state.live_feedback:
        fb_items = "".join([
            f'<div style="display:flex;align-items:flex-start;gap:10px;padding:8px 0;border-bottom:1px solid rgba(74,222,128,0.08);">'
            f'<span style="color:#fbbf24;font-size:14px;margin-top:1px;">→</span>'
            f'<span style="font-size:13px;color:#d1fae5;line-height:1.5;">{fb}</span>'
            f'</div>'
            for fb in st.session_state.live_feedback
        ])
        st.markdown(glass_card(f"""
            <p style="font-family:Syne,sans-serif;font-size:11px;font-weight:700;
              letter-spacing:0.12em;text-transform:uppercase;color:#fbbf24;opacity:0.8;margin-bottom:12px;">
              Corrective Feedback
            </p>
            {fb_items}
        """), unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
st.markdown("<br/>", unsafe_allow_html=True)
tab_session, tab_reps, tab_analytics, tab_guide = st.tabs([
    "  📊 Session  ", "  🔁 Rep Log  ", "  📈 Analytics  ", "  📖 Setup Guide  "
])

# ──────────────────────────────────────────────────────────────────────────────
# TAB 1 — Session Overview
# ──────────────────────────────────────────────────────────────────────────────
with tab_session:
    reps = st.session_state.reps
    scores = [r["score"] for r in reps]

    # KPI row
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Total Reps", len(reps))
    with k2:
        st.metric("Avg Score", f"{np.mean(scores):.0f}" if scores else "--")
    with k3:
        st.metric("Best Rep", max(scores) if scores else "--")
    with k4:
        dur = ""
        if st.session_state.session_start:
            secs = int((datetime.datetime.now() - st.session_state.session_start).total_seconds())
            dur  = f"{secs//60}:{secs%60:02d}"
        st.metric("Duration", dur or "--")

    st.markdown("<br/>", unsafe_allow_html=True)
    col_status, col_alerts = st.columns([1, 1], gap="large")

    with col_status:
        trend_label, slope = compute_trend(scores)
        overload = overload_ready(reps)
        freq = error_frequency(reps)

        st.markdown(glass_card(f"""
          <p style="font-family:Syne,sans-serif;font-size:11px;font-weight:700;
            letter-spacing:0.12em;text-transform:uppercase;color:#4ade80;opacity:0.7;margin-bottom:16px;">
            Session Status
          </p>
          <div style="display:flex;flex-direction:column;gap:14px;">
            <div>
              <p style="font-size:12px;color:#86efac;opacity:0.7;margin-bottom:6px;">Form Trend</p>
              {trend_badge(trend_label)}
              <span style="font-size:12px;color:#86efac;opacity:0.5;margin-left:8px;">
                {f'{slope:+.1f} pts/rep' if len(scores) >= 4 else ''}
              </span>
            </div>
            <div>
              <p style="font-size:12px;color:#86efac;opacity:0.7;margin-bottom:6px;">Progressive Overload</p>
              {'<span style="color:#4ade80;font-weight:600;font-size:13px;">✓ Ready to increase load</span>' if overload else
               '<span style="color:#9ca3af;font-size:13px;">3 reps ≥80 needed</span>'}
            </div>
            <div>
              <p style="font-size:12px;color:#86efac;opacity:0.7;margin-bottom:6px;">Top Error</p>
              <span style="font-size:13px;color:#fca5a5;">
                {list(freq.keys())[0].replace('_',' ').title() if freq else 'None detected'}
              </span>
            </div>
          </div>
        """), unsafe_allow_html=True)

    with col_alerts:
        fatigue = len(scores) >= 4 and trend_label == "declining"
        plateau = len(scores) >= 5 and (max(scores[-5:]) - min(scores[-5:])) <= 5

        alerts = []
        if fatigue:
            alerts.append(("⚡ Fatigue Detected", "Form declining — consider rest or load reduction.", "#f87171", "rgba(248,113,113,0.12)"))
        if overload:
            alerts.append(("🚀 Overload Ready", "3 consecutive reps ≥80 — safe to increase weight.", "#4ade80", "rgba(74,222,128,0.1)"))
        if plateau:
            alerts.append(("📊 Plateau Detected", "Scores stable — try technique cues or load change.", "#fbbf24", "rgba(251,191,36,0.1)"))
        if not alerts:
            alerts.append(("✓ All Clear", "No alerts — keep going!", "#86efac", "rgba(134,239,172,0.08)"))

        alerts_html = ""
        for title, msg, col, bg in alerts:
            alerts_html += f"""
            <div style="background:{bg};border:1px solid {col}44;border-radius:12px;
              padding:14px;margin-bottom:10px;">
              <p style="font-family:Syne,sans-serif;font-size:13px;font-weight:700;
                color:{col};margin-bottom:4px;">{title}</p>
              <p style="font-size:12px;color:#d1fae5;opacity:0.8;line-height:1.5;">{msg}</p>
            </div>"""

        st.markdown(glass_card(f"""
          <p style="font-family:Syne,sans-serif;font-size:11px;font-weight:700;
            letter-spacing:0.12em;text-transform:uppercase;color:#4ade80;opacity:0.7;margin-bottom:16px;">
            Alerts
          </p>
          {alerts_html}
        """), unsafe_allow_html=True)

    # Score bar chart
    if scores:
        st.markdown("<br/>", unsafe_allow_html=True)
        bar_items = ""
        for i, r in enumerate(reps):
            s = r["score"]
            col = "#4ade80" if s >= 75 else "#fbbf24" if s >= 50 else "#f87171"
            pct = s
            bar_items += f"""
            <div style="display:flex;flex-direction:column;align-items:center;gap:4px;flex:1;min-width:24px;">
              <div style="font-size:10px;color:{col};font-weight:600;">{s}</div>
              <div style="width:100%;background:rgba(20,83,45,0.4);border-radius:6px;
                height:80px;display:flex;align-items:flex-end;overflow:hidden;">
                <div style="width:100%;height:{pct}%;background:{col};
                  box-shadow:0 0 8px {col}55;border-radius:4px;transition:height 0.4s;"></div>
              </div>
              <div style="font-size:9px;color:#86efac;opacity:0.6;">R{r['rep_number']}</div>
            </div>"""

        st.markdown(glass_card(f"""
          <p style="font-family:Syne,sans-serif;font-size:11px;font-weight:700;
            letter-spacing:0.12em;text-transform:uppercase;color:#4ade80;opacity:0.7;margin-bottom:16px;">
            Rep Scores
          </p>
          <div style="display:flex;align-items:flex-end;gap:6px;height:110px;overflow-x:auto;padding-bottom:4px;">
            {bar_items}
          </div>
        """), unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# TAB 2 — Rep Log
# ──────────────────────────────────────────────────────────────────────────────
with tab_reps:
    reps = st.session_state.reps
    if not reps:
        st.markdown(glass_card("""
          <div style="text-align:center;padding:40px 0;">
            <div style="font-size:40px;opacity:0.3;margin-bottom:12px;">🔁</div>
            <p style="color:#4ade80;opacity:0.5;font-family:Syne,sans-serif;font-size:14px;">
              No reps logged yet — press "Simulate Rep" or start a live session.
            </p>
          </div>
        """), unsafe_allow_html=True)
    else:
        for rep in reversed(reps):
            s = rep["score"]
            col = "#4ade80" if s >= 75 else "#fbbf24" if s >= 50 else "#f87171"
            detected_errors = [e for e, v in rep["errors"].items() if v]
            pills_html = " ".join([pill(e, True) for e in detected_errors]) if detected_errors else \
                         '<span style="color:#4ade80;font-size:12px;">✓ Clean rep</span>'

            with st.expander(f"Rep {rep['rep_number']}  —  Score: {s}/100", expanded=(rep["rep_number"] == len(reps))):
                left, right = st.columns([1, 2])
                with left:
                    st.markdown(f"<div style='display:flex;justify-content:center;'>{score_ring(s, 100)}</div>",
                                unsafe_allow_html=True)
                with right:
                    st.markdown(f"""
                    <div style="padding-top:8px;">
                      <div style="display:flex;flex-wrap:wrap;gap:6px;margin-bottom:12px;">{pills_html}</div>
                      {''.join([f'<p style="font-size:13px;color:#fca5a5;margin:4px 0;">→ {fb}</p>' for fb in rep["feedback"]]) or
                       '<p style="font-size:13px;color:#86efac;">No corrective feedback needed.</p>'}
                      <p style="font-size:10px;color:#4ade80;opacity:0.4;margin-top:12px;">{rep.get("timestamp","")[:19]}</p>
                    </div>
                    """, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# TAB 3 — Analytics
# ──────────────────────────────────────────────────────────────────────────────
with tab_analytics:
    reps = st.session_state.reps
    freq = error_frequency(reps)
    scores = [r["score"] for r in reps]

    if not reps:
        st.info("Log some reps to see analytics.")
    else:
        left_a, right_a = st.columns(2, gap="large")

        with left_a:
            # Error frequency bars
            max_freq = max(freq.values()) if freq else 1
            freq_items = ""
            for err, cnt in freq.items():
                pct = int(cnt / max_freq * 100)
                freq_items += f"""
                <div style="margin-bottom:12px;">
                  <div style="display:flex;justify-content:space-between;margin-bottom:5px;">
                    <span style="font-size:13px;color:#d1fae5;">{err.replace('_',' ').title()}</span>
                    <span style="font-size:12px;color:#4ade80;font-weight:600;">{cnt}/{len(reps)}</span>
                  </div>
                  {mini_bar(pct, 100, "#f87171" if pct > 50 else "#fbbf24" if pct > 25 else "#4ade80")}
                </div>"""

            st.markdown(glass_card(f"""
              <p style="font-family:Syne,sans-serif;font-size:11px;font-weight:700;
                letter-spacing:0.12em;text-transform:uppercase;color:#4ade80;opacity:0.7;margin-bottom:16px;">
                Error Frequency
              </p>
              {freq_items if freq_items else '<p style="color:#4ade80;opacity:0.5;font-size:13px;">No errors detected.</p>'}
            """), unsafe_allow_html=True)

        with right_a:
            # Score distribution
            if scores:
                buckets = {"≥80 (Good)": 0, "60–79 (Ok)": 0, "<60 (Poor)": 0}
                for s in scores:
                    if s >= 80:   buckets["≥80 (Good)"] += 1
                    elif s >= 60: buckets["60–79 (Ok)"] += 1
                    else:         buckets["<60 (Poor)"] += 1
                total = len(scores)
                dist_items = ""
                bcolors = {"≥80 (Good)": "#4ade80", "60–79 (Ok)": "#fbbf24", "<60 (Poor)": "#f87171"}
                for label, cnt in buckets.items():
                    pct = int(cnt / total * 100) if total else 0
                    dist_items += f"""
                    <div style="margin-bottom:12px;">
                      <div style="display:flex;justify-content:space-between;margin-bottom:5px;">
                        <span style="font-size:13px;color:#d1fae5;">{label}</span>
                        <span style="font-size:12px;font-weight:600;color:{bcolors[label]};">{cnt} ({pct}%)</span>
                      </div>
                      {mini_bar(pct, 100, bcolors[label])}
                    </div>"""
                st.markdown(glass_card(f"""
                  <p style="font-family:Syne,sans-serif;font-size:11px;font-weight:700;
                    letter-spacing:0.12em;text-transform:uppercase;color:#4ade80;opacity:0.7;margin-bottom:16px;">
                    Score Distribution
                  </p>
                  {dist_items}
                """), unsafe_allow_html=True)

        st.markdown("<br/>", unsafe_allow_html=True)

        # Coaching tip
        if freq:
            top_err = list(freq.keys())[0]
            tips = {
                "knees_inward":  "Drill: 3×12 banded squats to cue knee tracking outward.",
                "rounded_back":  "Drill: Paused Romanian deadlift to reinforce neutral spine.",
                "shallow_squat": "Drill: Box squat to a low box — build depth confidence.",
                "knees_forward": "Drill: Wall squat — toes 2cm from wall.",
                "elbow_error":   "Drill: Z-press (seated floor press) to isolate vertical path.",
                "lumbar_error":  "Drill: Chest-supported row to remove spinal loading.",
                "asymmetry":     "Drill: Single-arm rows to correct left-right imbalance.",
                "depth_error":   "Drill: Paused rows at contraction point.",
                "knees_error":   "Drill: Tempo OHP 3-1-1 — slow eccentric for control.",
                "spine_error":   "Drill: Dead-stop OHP from rack pins at forehead height.",
            }
            tip = tips.get(top_err, "Review your form with a qualified coach.")
            st.markdown(glass_card(f"""
              <p style="font-family:Syne,sans-serif;font-size:11px;font-weight:700;
                letter-spacing:0.12em;text-transform:uppercase;color:#fbbf24;opacity:0.8;margin-bottom:12px;">
                🎯 Top Coaching Recommendation
              </p>
              <p style="font-size:13px;color:#d1fae5;line-height:1.6;">
                Most frequent error: <strong style="color:#fca5a5;">{top_err.replace('_',' ').title()}</strong><br/>
                {tip}
              </p>
            """), unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# TAB 4 — Setup Guide
# ──────────────────────────────────────────────────────────────────────────────
with tab_guide:
    steps = [
        ("01", "Environment Setup",
         "python -m venv venv && source venv/bin/activate\npip install mediapipe opencv-python numpy scipy scikit-learn torch streamlit",
         "Python 3.10+ required. GPU optional — MediaPipe runs on CPU."),
        ("02", "Project Structure",
         "gym_aq/\n├── app.py          ← this Streamlit frontend\n├── main.py         ← CLI live webcam mode\n├── modules/\n│   ├── extractor.py   ← MediaPipe pose\n│   ├── segmentor.py   ← rep detection\n│   ├── features.py    ← biomechanics\n│   ├── classifier.py  ← SVM error detection\n│   └── tracker.py     ← session analytics\n├── models/            ← trained .pkl files\n└── logs/              ← session JSON exports",
         "Each module is independently importable."),
        ("03", "Download Fitness-AQA Dataset",
         "# Request access at:\n# https://github.com/ParitoshParmar/Fitness-AQA\n# Place clips in: gym_aq/data/",
         "11K+ annotated clips · BackSquat · OHP · Barbell Row"),
        ("04", "Train Classifiers",
         "from modules.classifier import FormClassifier\nfrom modules.features import compute_rep_features\n\nclf = FormClassifier('BackSquat')\nclf.train(X_train, y_dict)  # from Fitness-AQA\nclf.save('models/backsquat_svm.pkl')",
         "One SVM per error class · RBF kernel · MinMax scaled."),
        ("05", "Run Streamlit App",
         "streamlit run app.py",
         "Opens at http://localhost:8501 · Use demo mode without a trained model."),
        ("06", "Run Live CLI Mode",
         "python main.py --exercise BackSquat --athlete lionel",
         "Webcam feed with real-time skeleton overlay · press Q to quit."),
    ]

    for num, title, code, note in steps:
        st.markdown(glass_card(f"""
          <div style="display:flex;gap:16px;align-items:flex-start;">
            <div style="flex-shrink:0;width:36px;height:36px;border-radius:10px;
              background:rgba(34,197,94,0.15);border:1px solid rgba(74,222,128,0.3);
              display:flex;align-items:center;justify-content:center;
              font-family:Syne,sans-serif;font-size:12px;font-weight:800;color:#4ade80;">
              {num}
            </div>
            <div style="flex:1;min-width:0;">
              <p style="font-family:Syne,sans-serif;font-size:14px;font-weight:700;
                color:#ecfdf5;margin-bottom:8px;">{title}</p>
              <pre style="background:rgba(5,14,8,0.7);border:1px solid rgba(74,222,128,0.12);
                border-radius:10px;padding:12px;font-size:12px;color:#86efac;
                overflow-x:auto;margin-bottom:8px;font-family:monospace;line-height:1.6;">{code}</pre>
              <p style="font-size:12px;color:#4ade80;opacity:0.6;">{note}</p>
            </div>
          </div>
        """, "margin-bottom:12px;"), unsafe_allow_html=True)

# ── Auto-refresh when running ─────────────────────────────────────────────────
if st.session_state.running:
    time.sleep(0.1)
    st.rerun()
