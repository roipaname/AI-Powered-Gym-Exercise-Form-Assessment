"""
app.py — FormIQ Streamlit Dashboard
Run: streamlit run app.py
"""

import streamlit as st
import numpy as np
import json
import time
import random
import datetime
import threading
import av
import cv2
from pathlib import Path

try:
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False

# Suppress MediaPipe logs
import os
os.environ["GLOG_minloglevel"]      = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["MEDIAPIPE_DISABLE_GPU"]= "1"
import warnings; warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="FormIQ",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Live feed reader ──────────────────────────────────────────────────────────
import base64
from pathlib import Path as _Path

LIVE_DIR   = _Path(".formiq_live")
FRAME_FILE = LIVE_DIR / "frame.jpg"
STATE_FILE = LIVE_DIR / "state.json"

def read_live_frame():
    """Returns base64 JPEG string if main.py is running, else None."""
    try:
        if FRAME_FILE.exists() and FRAME_FILE.stat().st_mtime > time.time() - 3:
            return base64.b64encode(FRAME_FILE.read_bytes()).decode()
    except Exception:
        pass
    return None

def read_live_state():
    """Returns state dict from main.py or None."""
    try:
        if STATE_FILE.exists() and STATE_FILE.stat().st_mtime > time.time() - 3:
            return json.loads(STATE_FILE.read_text())
    except Exception:
        pass
    return None

# ══════════════════════════════════════════════════════════════════════════════
# DESIGN SYSTEM
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Bebas+Neue&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  /* Core palette — deep teal-to-lime */
  --ink:     #03080a;
  --ink2:    #07141a;
  --surface: rgba(5,20,28,0.72);
  --rim:     rgba(0,230,160,0.14);
  --rim2:    rgba(0,230,160,0.28);

  /* Gradient accents */
  --g-lime:  #a8ff3e;
  --g-teal:  #00e6a0;
  --g-cyan:  #00c8e0;
  --g-amber: #ffb800;
  --g-red:   #ff4d6d;

  /* Text */
  --tx:      #e8f5ee;
  --tx2:     #7bbf9a;
  --tx3:     #3d7055;

  /* Gradients */
  --grad-main: linear-gradient(135deg, #a8ff3e 0%, #00e6a0 50%, #00c8e0 100%);
  --grad-warm: linear-gradient(135deg, #ffb800 0%, #ff4d6d 100%);
  --grad-glass: linear-gradient(135deg,rgba(0,230,160,0.08),rgba(0,200,224,0.04));
}

/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; }
html, body, [data-testid="stAppViewContainer"],
[data-testid="stApp"], .main, .block-container {
  background: transparent !important;
}
[data-testid="stApp"]::before {
  content:''; position:fixed; inset:0; z-index:-2;
  background:
    radial-gradient(ellipse 70% 60% at 15% 10%, rgba(0,180,120,0.12) 0%, transparent 65%),
    radial-gradient(ellipse 50% 70% at 85% 85%, rgba(0,150,200,0.10) 0%, transparent 60%),
    radial-gradient(ellipse 80% 80% at 50% 50%, #03080a 0%, #010608 100%);
}
/* Fine dot grid */
[data-testid="stApp"]::after {
  content:''; position:fixed; inset:0; z-index:-1; pointer-events:none;
  background-image: radial-gradient(circle, rgba(0,230,160,0.06) 1px, transparent 1px);
  background-size: 32px 32px;
}

/* ── Typography ── */
body, p, span, div, label { font-family:'Space Grotesk',sans-serif !important; color:var(--tx) !important; }
h1,h2,h3,h4 { font-family:'Bebas Neue',sans-serif !important; letter-spacing:0.06em; color:var(--tx) !important; }
code, pre { font-family:'JetBrains Mono',monospace !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: rgba(3,10,12,0.88) !important;
  border-right: 1px solid var(--rim) !important;
  backdrop-filter: blur(28px) !important;
}
[data-testid="stSidebar"] * { color:var(--tx) !important; font-family:'Space Grotesk',sans-serif !important; }

/* ── Selectbox ── */
.stSelectbox>div>div {
  background:rgba(0,230,160,0.06) !important;
  border:1px solid var(--rim) !important;
  border-radius:10px !important;
}
.stSelectbox [data-baseweb="select"]>div { background:transparent !important; border:none !important; }

/* ── Text input ── */
.stTextInput>div>div>input {
  background:rgba(0,230,160,0.05) !important;
  border:1px solid var(--rim) !important;
  border-radius:10px !important;
  color:var(--tx) !important;
}

/* ── Buttons ── */
.stButton>button {
  background:var(--grad-glass) !important;
  border:1px solid var(--rim) !important;
  border-radius:10px !important;
  color:var(--g-teal) !important;
  font-family:'Space Grotesk',sans-serif !important;
  font-weight:600 !important;
  font-size:13px !important;
  letter-spacing:0.04em !important;
  padding:9px 18px !important;
  transition:all 0.18s ease !important;
  width:100% !important;
}
.stButton>button:hover {
  background:rgba(0,230,160,0.14) !important;
  border-color:var(--rim2) !important;
  box-shadow:0 0 20px rgba(0,230,160,0.18) !important;
  color:#fff !important;
  transform:translateY(-1px);
}

/* ── Download button ── */
.stDownloadButton>button {
  background:rgba(0,230,160,0.1) !important;
  border:1px solid var(--rim2) !important;
  border-radius:10px !important;
  color:var(--g-lime) !important;
  font-family:'Space Grotesk',sans-serif !important;
  font-weight:600 !important;
  width:100% !important;
}

/* ── Metrics ── */
[data-testid="metric-container"] {
  background:var(--grad-glass) !important;
  border:1px solid var(--rim) !important;
  border-radius:14px !important;
  padding:18px !important;
  backdrop-filter:blur(12px) !important;
  transition:border-color 0.2s;
}
[data-testid="metric-container"]:hover { border-color:var(--rim2) !important; }
[data-testid="stMetricValue"] {
  font-family:'Bebas Neue',sans-serif !important;
  font-size:2.4rem !important;
  background:var(--grad-main);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent;
  background-clip:text;
}
[data-testid="stMetricLabel"] {
  font-size:11px !important; text-transform:uppercase !important;
  letter-spacing:0.12em !important; color:var(--tx3) !important;
  font-weight:600 !important;
}
[data-testid="stMetricDelta"] svg { display:none; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
  background:rgba(0,230,160,0.04) !important;
  border:1px solid var(--rim) !important;
  border-radius:12px !important;
  padding:4px !important;
  gap:3px !important;
}
.stTabs [data-baseweb="tab"] {
  background:transparent !important;
  color:var(--tx3) !important;
  font-family:'Space Grotesk',sans-serif !important;
  font-weight:600 !important;
  font-size:12px !important;
  letter-spacing:0.06em !important;
  border-radius:9px !important;
  padding:7px 18px !important;
  border:none !important;
  text-transform:uppercase !important;
}
.stTabs [aria-selected="true"] {
  background:rgba(0,230,160,0.12) !important;
  color:var(--g-teal) !important;
  box-shadow:0 0 14px rgba(0,230,160,0.15) !important;
}
.stTabs [data-baseweb="tab-border"] { display:none !important; }

/* ── Progress ── */
.stProgress>div>div>div {
  background:var(--grad-main) !important;
  border-radius:99px !important;
  box-shadow:0 0 10px rgba(0,230,160,0.3) !important;
}
.stProgress>div>div { background:rgba(0,230,160,0.08) !important; border-radius:99px !important; }

/* ── Expander ── */
.streamlit-expanderHeader {
  background:rgba(0,230,160,0.05) !important;
  border:1px solid var(--rim) !important;
  border-radius:10px !important;
  font-family:'Space Grotesk',sans-serif !important;
}

/* ── Alert ── */
.stAlert { background:rgba(0,230,160,0.06) !important; border:1px solid var(--rim) !important; border-radius:12px !important; }

/* ── Divider ── */
hr { border-color:var(--rim) !important; margin:14px 0 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width:5px; height:5px; }
::-webkit-scrollbar-track { background:transparent; }
::-webkit-scrollbar-thumb { background:var(--tx3); border-radius:99px; }

/* ── Hide chrome ── */
#MainMenu, footer, header, [data-testid="stToolbar"] { visibility:hidden; }
.block-container { padding-top:1.8rem !important; }

/* ── Pulse animation ── */
@keyframes pulse { 0%,100%{opacity:1;} 50%{opacity:0.35;} }
@keyframes fadeIn { from{opacity:0;transform:translateY(6px);} to{opacity:1;transform:translateY(0);} }
.fade-in { animation: fadeIn 0.4s ease forwards; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# COMPONENT LIBRARY
# ══════════════════════════════════════════════════════════════════════════════

def card(html, extra=""):
    return (
        '<div style="background:linear-gradient(135deg,rgba(0,230,160,0.07),rgba(0,200,224,0.04));'
        'border:1px solid rgba(0,230,160,0.14);border-radius:16px;padding:22px;'
        'backdrop-filter:blur(18px);-webkit-backdrop-filter:blur(18px);'
        + extra + '">' + html + '</div>'
    )


def score_ring_svg(score, size=128):
    pct   = max(0, min(100, score if isinstance(score, int) else 0))
    r     = 44
    circ  = 2 * 3.14159 * r
    dash  = circ * pct / 100
    offset= circ / 4
    if pct >= 75:   col, glow = "#a8ff3e", "rgba(168,255,62,0.5)"
    elif pct >= 50: col, glow = "#ffb800", "rgba(255,184,0,0.5)"
    else:           col, glow = "#ff4d6d", "rgba(255,77,109,0.5)"
    label = str(score) if isinstance(score, int) else "--"
    return f"""
    <svg width="{size}" height="{size}" viewBox="0 0 100 100">
      <circle cx="50" cy="50" r="{r}" fill="none" stroke="rgba(0,80,50,0.5)" stroke-width="7"/>
      <circle cx="50" cy="50" r="{r}" fill="none" stroke="{col}" stroke-width="7"
        stroke-dasharray="{dash:.1f} {circ:.1f}" stroke-dashoffset="{offset:.1f}"
        stroke-linecap="round" style="filter:drop-shadow(0 0 7px {glow});
        transition:stroke-dasharray 0.6s cubic-bezier(0.4,0,0.2,1);"/>
      <text x="50" y="46" text-anchor="middle" font-size="21" font-weight="700"
        font-family="Bebas Neue,sans-serif" fill="{col}" letter-spacing="1">{label}</text>
      <text x="50" y="60" text-anchor="middle" font-size="8"
        font-family="Space Grotesk,sans-serif" fill="rgba(123,191,154,0.6)">/100</text>
    </svg>"""


def pill(label, detected):
    if detected:
        bg,bd,col,ic = "rgba(255,77,109,0.12)","rgba(255,77,109,0.4)","#ff8fa3","✕"
    else:
        bg,bd,col,ic = "rgba(0,230,160,0.08)","rgba(0,230,160,0.25)","#00e6a0","✓"
    return f"""<span style="background:{bg};border:1px solid {bd};color:{col};
      font-family:Space Grotesk,sans-serif;font-size:11px;font-weight:600;
      padding:3px 11px;border-radius:99px;display:inline-flex;align-items:center;
      gap:5px;letter-spacing:0.03em;white-space:nowrap;">{ic} {label.replace('_',' ')}</span>"""


def trend_chip(label):
    m = {
        "improving":         ("↗","#a8ff3e","rgba(168,255,62,0.1)","rgba(168,255,62,0.28)"),
        "stable":            ("→","#00e6a0","rgba(0,230,160,0.08)","rgba(0,230,160,0.22)"),
        "declining":         ("↘","#ff4d6d","rgba(255,77,109,0.1)","rgba(255,77,109,0.3)"),
        "insufficient data": ("·","#3d7055","rgba(61,112,85,0.1)","rgba(61,112,85,0.2)"),
    }
    ic,col,bg,bd = m.get(label, m["insufficient data"])
    return f"""<span style="background:{bg};border:1px solid {bd};color:{col};
      font-family:Bebas Neue,sans-serif;font-size:15px;letter-spacing:0.08em;
      padding:4px 14px;border-radius:99px;display:inline-flex;align-items:center;gap:7px;">
      {ic} {label.upper()}</span>"""


def mini_bar(pct, color="#00e6a0"):
    return f"""<div style="width:100%;height:5px;background:rgba(0,100,60,0.3);border-radius:99px;overflow:hidden;margin-top:5px;">
      <div style="width:{min(100,pct)}%;height:100%;background:{color};border-radius:99px;
        box-shadow:0 0 8px {color}44;"></div></div>"""


def section_label(text):
    return f"""<p style="font-family:Space Grotesk,sans-serif;font-size:10px;font-weight:700;
      letter-spacing:0.14em;text-transform:uppercase;color:#3d7055;margin-bottom:10px;">{text}</p>"""


def alert_box(icon, title, msg, col, bg):
    return f"""<div style="background:{bg};border:1px solid {col}55;border-radius:12px;
      padding:12px 14px;margin-bottom:8px;">
      <p style="font-family:Bebas Neue,sans-serif;font-size:15px;letter-spacing:0.06em;
        color:{col};margin-bottom:3px;">{icon} {title}</p>
      <p style="font-size:12px;color:rgba(232,245,238,0.75);line-height:1.5;">{msg}</p>
    </div>"""


# ══════════════════════════════════════════════════════════════════════════════
# STATE
# ══════════════════════════════════════════════════════════════════════════════

def init():
    for k, v in {
        "reps": [], "running": False, "exercise": "BackSquat",
        "athlete_id": "Athlete", "session_start": None,
        "live_score": 0, "live_errors": {}, "live_feedback": [],
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v

init()


# ══════════════════════════════════════════════════════════════════════════════
# LIVE VIDEO PROCESSOR  (streamlit-webrtc)
# ══════════════════════════════════════════════════════════════════════════════

MODEL_MAP = {
    "BackSquat":    "models/backsquat_svm.pkl",
    "OverheadPress":"models/overheadpress_svm.pkl",
    "BarbellRow":   "models/barbellrow_svm.pkl",
}

if WEBRTC_AVAILABLE:
    class FormVideoProcessor(VideoProcessorBase):
        """
        Runs in the webrtc thread.
        Extracts pose → segments reps → classifies errors each frame.
        Writes results to session_state so the main thread can read them.
        """
        C_LIME  = (62,  255, 168)
        C_AMBER = (40,  184, 255)
        C_RED   = (109, 77,  255)
        C_MUTED = (85,  112, 61)
        C_DARK  = (8,   20,  3)

        def __init__(self):
            from modules.extractor  import PoseExtractor
            from modules.segmentor  import RepSegmentor
            from modules.classifier import FormClassifier
            self.extractor  = PoseExtractor(min_detection_conf=0.4,
                                            min_tracking_conf=0.4,
                                            model_complexity=1)
            self.segmentor  = RepSegmentor()
            self.classifier = FormClassifier(st.session_state.exercise)
            mp = MODEL_MAP.get(st.session_state.exercise, "")
            if mp and Path(mp).exists():
                self.classifier.load(mp)
            self.lm_buf         = []
            self.last_rep_count = 0
            self.lock           = threading.Lock()
            self.score          = 0
            self.errors         = {}
            self.feedback       = []
            self.rep_count      = 0

        def _score_col(self, s):
            if s >= 75: return self.C_LIME
            if s >= 50: return self.C_AMBER
            return self.C_RED

        def _draw_hud(self, frame, h, w):
            pw = min(240, w // 3)
            ov = frame.copy()
            cv2.rectangle(ov, (0,0), (pw,h), self.C_DARK, -1)
            cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)
            cv2.rectangle(frame, (0,0), (pw,h), (55,160,0), 1, cv2.LINE_AA)
            # gradient title bar
            for i in range(pw):
                t = i / pw
                cv2.line(frame,(i,0),(i,30),(int(3+t*10),int(180+t*75),int(60+t*108)),1)
            cv2.putText(frame,"FORMIQ",(8,21),cv2.FONT_HERSHEY_SIMPLEX,0.6,(5,15,5),2,cv2.LINE_AA)
            # score arc
            cx,cy,r = 48,86,34
            sc  = self.score if isinstance(self.score,int) else 0
            col = self._score_col(sc)
            cv2.ellipse(frame,(cx,cy),(r,r),-90,0,360,(20,60,20),5,cv2.LINE_AA)
            if sc>0: cv2.ellipse(frame,(cx,cy),(r,r),-90,0,int(360*sc/100),col,5,cv2.LINE_AA)
            lbl = str(sc)
            tw  = cv2.getTextSize(lbl,cv2.FONT_HERSHEY_SIMPLEX,0.75,2)[0][0]
            cv2.putText(frame,lbl,(cx-tw//2,cy+6),cv2.FONT_HERSHEY_SIMPLEX,0.75,col,2,cv2.LINE_AA)
            cv2.putText(frame,"/100",(cx-12,cy+18),cv2.FONT_HERSHEY_SIMPLEX,0.27,self.C_MUTED,1,cv2.LINE_AA)
            # reps
            cv2.putText(frame,"REPS",(100,68),cv2.FONT_HERSHEY_SIMPLEX,0.3,self.C_MUTED,1,cv2.LINE_AA)
            cv2.putText(frame,str(self.rep_count),(100,100),cv2.FONT_HERSHEY_SIMPLEX,1.0,(220,235,220),2,cv2.LINE_AA)
            cv2.line(frame,(6,122),(pw-6,122),(40,100,20),1)
            y = 140
            if self.errors:
                cv2.putText(frame,"ERRORS",(6,y-4),cv2.FONT_HERSHEY_SIMPLEX,0.27,self.C_MUTED,1)
                for err,det in self.errors.items():
                    c = self.C_RED if det else self.C_LIME
                    cv2.putText(frame,f"{'X' if det else 'OK'} {err.replace('_',' ')}",(6,y+12),
                                cv2.FONT_HERSHEY_SIMPLEX,0.3,c,1,cv2.LINE_AA)
                    y += 18
            if self.feedback and y < h-40:
                cv2.line(frame,(6,y+3),(pw-6,y+3),(40,100,20),1)
                y += 14
                for fb in self.feedback[:2]:
                    # word-wrap inside panel
                    words = fb.split(); line = ""
                    for wd in words:
                        test = (line+" "+wd).strip()
                        if cv2.getTextSize(test,cv2.FONT_HERSHEY_SIMPLEX,0.3,1)[0][0] < pw-12:
                            line = test
                        else:
                            cv2.putText(frame,line,(6,y),cv2.FONT_HERSHEY_SIMPLEX,0.3,self.C_AMBER,1,cv2.LINE_AA)
                            y+=14; line=wd
                    if line:
                        cv2.putText(frame,line,(6,y),cv2.FONT_HERSHEY_SIMPLEX,0.3,self.C_AMBER,1,cv2.LINE_AA)
                        y+=14
            # right-edge flash
            cv2.rectangle(frame,(w-4,0),(w,h),self._score_col(sc),-1)

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            h, w = img.shape[:2]
            rgb    = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            lm,ann = self.extractor.extract(rgb)
            display= cv2.cvtColor(ann, cv2.COLOR_RGB2BGR)
            if lm is not None:
                self.segmentor.update(lm)
                self.lm_buf.append(lm)
                reps = self.segmentor.get_reps()
                if len(reps) > self.last_rep_count:
                    rep = reps[-1]
                    seq = self.lm_buf[rep.start_frame:rep.end_frame+1]
                    if len(seq) > 3:
                        from modules.features import compute_rep_features
                        feats = compute_rep_features(seq)
                        errors,score,feedback = self.classifier.predict(feats)
                        with self.lock:
                            self.score     = score
                            self.errors    = errors
                            self.feedback  = feedback
                            self.rep_count = len(reps)
                        # push to session state (thread-safe for simple writes)
                        st.session_state.live_score    = score
                        st.session_state.live_errors   = errors
                        st.session_state.live_feedback = feedback
                        st.session_state.reps.append({
                            "rep_number": len(st.session_state.reps)+1,
                            "score":      score,
                            "errors":     errors,
                            "feedback":   feedback,
                            "timestamp":  datetime.datetime.now().isoformat(),
                        })
                    self.last_rep_count = len(reps)
            self._draw_hud(display, h, w)
            return av.VideoFrame.from_ndarray(display, format="bgr24")

        def __del__(self):
            try: self.extractor.close()
            except: pass


# ══════════════════════════════════════════════════════════════════════════════
# DEMO ENGINE
# ══════════════════════════════════════════════════════════════════════════════

EXERCISE_ERRORS = {
    "BackSquat":     ["knees_inward","knees_forward","rounded_back","shallow_squat"],
    "OverheadPress": ["elbow_error","knees_error","spine_error"],
    "BarbellRow":    ["lumbar_error","torso_angle_error"],
}
FEEDBACK = {
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

def sim_rep(exercise, n, fatigue=False):
    errs   = EXERCISE_ERRORS[exercise]
    base   = max(52, 94 - n*(3 if fatigue else 0.4) + random.gauss(0,4))
    errors = {e: random.random() < (0.12 + n*0.018) for e in errs}
    score  = max(18, min(100, int(base - sum(errors.values())*11)))
    return {"rep_number":n,"score":score,"errors":errors,
            "feedback":[FEEDBACK[e] for e,v in errors.items() if v],
            "timestamp":datetime.datetime.now().isoformat()}

def trend_calc(scores):
    if len(scores) < 4: return "insufficient data", 0.0
    s = float(np.polyfit(range(len(scores)), scores, 1)[0])
    return ("declining" if s < -3 else "improving" if s > 2 else "stable"), s

def err_freq(reps):
    f = {}
    for r in reps:
        for e, v in r["errors"].items():
            if v: f[e] = f.get(e,0)+1
    return dict(sorted(f.items(), key=lambda x:-x[1]))

def overload_ok(reps):
    return len(reps) >= 3 and all(r["score"] >= 80 for r in reps[-3:])


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    # Logo
    st.markdown("""
    <div style="padding:12px 0 20px;">
      <div style="display:flex;align-items:baseline;gap:6px;">
        <span style="font-family:Bebas Neue,sans-serif;font-size:32px;letter-spacing:0.1em;
          background:linear-gradient(135deg,#a8ff3e,#00e6a0);
          -webkit-background-clip:text;-webkit-text-fill-color:transparent;
          background-clip:text;">FORMIQ</span>
        <span style="font-size:10px;font-weight:700;letter-spacing:0.16em;
          color:#3d7055;text-transform:uppercase;padding-bottom:4px;">AI Coach</span>
      </div>
    </div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown(section_label("Session Config"), unsafe_allow_html=True)

    ex = st.selectbox("Exercise",
                      ["BackSquat","OverheadPress","BarbellRow"],
                      index=["BackSquat","OverheadPress","BarbellRow"]
                      .index(st.session_state.exercise))
    st.session_state.exercise = ex

    athlete = st.text_input("Athlete ID", value=st.session_state.athlete_id)
    st.session_state.athlete_id = athlete

    st.markdown(section_label("Camera"), unsafe_allow_html=True)
    st.selectbox("Source", ["Webcam 0","Webcam 1","IP Stream"])

    st.markdown("<br/>", unsafe_allow_html=True)
    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        if st.button("▶ START"):
            st.session_state.running      = True
            st.session_state.session_start = datetime.datetime.now()
    with c2:
        if st.button("■ STOP"):
            st.session_state.running = False

    if st.button("↺ RESET"):
        st.session_state.reps         = []
        st.session_state.running      = False
        st.session_state.live_score   = 0
        st.session_state.live_errors  = {}
        st.session_state.live_feedback= []
        st.session_state.session_start= None
        st.rerun()

    st.markdown("<br/>", unsafe_allow_html=True)
    if st.button("＋ SIMULATE REP"):
        n   = len(st.session_state.reps)+1
        rec = sim_rep(st.session_state.exercise, n)
        st.session_state.reps.append(rec)
        st.session_state.live_score    = rec["score"]
        st.session_state.live_errors   = rec["errors"]
        st.session_state.live_feedback = rec["feedback"]
        st.rerun()

    st.divider()
    st.markdown(section_label("Export"), unsafe_allow_html=True)
    if st.session_state.reps:
        st.download_button("⬇ DOWNLOAD JSON",
            data=json.dumps({"exercise":ex,"athlete_id":athlete,
                             "reps":st.session_state.reps}, indent=2),
            file_name=f"formiq_{ex}_{datetime.date.today()}.json",
            mime="application/json", use_container_width=True)

    st.markdown("<br/>", unsafe_allow_html=True)
    live = st.session_state.running
    st.markdown(f"""<div style="display:flex;align-items:center;gap:9px;padding:10px 12px;
      background:rgba(0,230,160,0.05);border:1px solid rgba(0,230,160,0.12);border-radius:10px;">
      <span style="width:7px;height:7px;border-radius:50%;
        background:{'#a8ff3e' if live else '#3d7055'};
        box-shadow:{'0 0 8px #a8ff3e' if live else 'none'};
        {'animation:pulse 1.2s infinite;' if live else ''}"></span>
      <span style="font-family:Bebas Neue,sans-serif;font-size:14px;letter-spacing:0.12em;
        color:{'#a8ff3e' if live else '#3d7055'};">{'LIVE' if live else 'IDLE'}</span>
    </div>
    <style>@keyframes pulse{{0%,100%{{opacity:1}}50%{{opacity:0.3}}}}</style>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ══════════════════════════════════════════════════════════════════════════════

# Header
st.markdown(f"""
<div style="margin-bottom:28px;">
  <h1 style="font-size:clamp(32px,5vw,52px);margin:0;line-height:1;
    background:linear-gradient(135deg,#a8ff3e 0%,#00e6a0 45%,#00c8e0 100%);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">
    EXERCISE FORM ASSESSMENT
  </h1>
  <p style="color:#3d7055;font-size:13px;margin-top:6px;letter-spacing:0.05em;
    font-family:Space Grotesk,sans-serif;">
    {st.session_state.exercise}&ensp;·&ensp;{st.session_state.athlete_id}&ensp;·&ensp;{datetime.date.today().strftime('%d %b %Y')}
  </p>
</div>""", unsafe_allow_html=True)

# ── Top row: camera + live score ──────────────────────────────────────────────
cam_col, score_col = st.columns([1.5, 1], gap="large")

with cam_col:
    if WEBRTC_AVAILABLE:
        # ── Live webrtc feed with full AI pipeline ────────────────────────
        st.markdown("""<style>
          .webrtc-wrap video {border-radius:14px !important; width:100% !important;}
          .webrtc-wrap {border-radius:14px;overflow:hidden;
            border:1px solid rgba(0,230,160,0.25);background:#020a0e;}
          .webrtc-wrap button {
            background:linear-gradient(135deg,rgba(0,230,160,0.18),rgba(0,200,224,0.12)) !important;
            border:1px solid rgba(0,230,160,0.35) !important;
            color:#a8ff3e !important;
            font-family:'Space Grotesk',sans-serif !important;
            font-weight:700 !important; letter-spacing:0.06em !important;
            border-radius:10px !important; padding:8px 20px !important;
          }
        </style>""", unsafe_allow_html=True)

        st.markdown('<div class="webrtc-wrap">', unsafe_allow_html=True)
        RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
        webrtc_streamer(
            key=f"formiq-{st.session_state.exercise}",
            video_processor_factory=FormVideoProcessor,
            rtc_configuration=RTC_CONFIG,
            media_stream_constraints={"video": {"width": {"ideal": 1280},
                                                "height": {"ideal": 720}}, "audio": False},
            async_processing=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<p style="font-size:11px;color:#3d7055;text-align:center;margin-top:6px;'
                    'font-family:Space Grotesk;letter-spacing:0.05em;">'
                    'MediaPipe · 33 keypoints · Real-time skeleton + HUD overlay</p>',
                    unsafe_allow_html=True)
    else:
        # ── Fallback placeholder when webrtc not installed ────────────────
        st.markdown("""<div style="aspect-ratio:16/9;background:rgba(2,10,14,0.9);
          border-radius:14px;border:1px solid rgba(0,230,160,0.1);
          display:flex;flex-direction:column;align-items:center;justify-content:center;gap:10px;">
          <p style="font-family:Bebas Neue,sans-serif;font-size:16px;letter-spacing:0.12em;
            color:#3d7055;margin:0;">INSTALL REQUIRED</p>
          <code style="background:rgba(0,230,160,0.08);color:#00e6a0;padding:4px 12px;
            border-radius:8px;font-size:12px;">pip install streamlit-webrtc aiortc</code>
          <p style="font-size:11px;color:#3d7055;text-align:center;">
            Then restart: streamlit run app.py
          </p>
        </div>""", unsafe_allow_html=True)

with score_col:
    score      = st.session_state.live_score
    ring_html  = score_ring_svg(score, 120)
    slabel     = section_label("Last Rep Score")
    pills_html = " ".join([pill(e, v) for e, v in st.session_state.live_errors.items()]) \
                 if st.session_state.live_errors else \
                 '<span style="color:#3d7055;font-size:12px;">No rep data yet</span>'

    score_card_html = (
        '<div style="background:linear-gradient(135deg,rgba(0,230,160,0.07),rgba(0,200,224,0.04));'
        'border:1px solid rgba(0,230,160,0.14);border-radius:16px;padding:22px;'
        'backdrop-filter:blur(18px);margin-bottom:14px;">'
        + slabel
        + '<div style="display:flex;justify-content:center;margin:8px 0 16px;">'
        + ring_html
        + '</div>'
        + '<div style="display:flex;flex-wrap:wrap;gap:6px;">'
        + pills_html
        + '</div></div>'
    )
    st.markdown(score_card_html, unsafe_allow_html=True)

    if st.session_state.live_feedback:
        fb_items = "".join([
            '<div style="display:flex;gap:9px;padding:7px 0;border-bottom:1px solid rgba(0,230,160,0.07);">'
            '<span style="color:#ffb800;font-size:13px;flex-shrink:0;">→</span>'
            f'<span style="font-size:12px;color:rgba(232,245,238,0.82);line-height:1.5;">{fb}</span>'
            '</div>'
            for fb in st.session_state.live_feedback
        ])
        fb_card = (
            '<div style="background:linear-gradient(135deg,rgba(0,230,160,0.07),rgba(0,200,224,0.04));'
            'border:1px solid rgba(0,230,160,0.14);border-radius:16px;padding:22px;'
            'backdrop-filter:blur(18px);">'
            + section_label("Corrective Feedback")
            + fb_items
            + '</div>'
        )
        st.markdown(fb_card, unsafe_allow_html=True)


# ── Tabs ──────────────────────────────────────────────────────────────────────
st.markdown("<br/>", unsafe_allow_html=True)
t1, t2, t3, t4 = st.tabs(["📊  Session","🔁  Rep Log","📈  Analytics","📖  Setup"])


# ════════════════════════════════════════════════
# TAB 1 — Session
# ════════════════════════════════════════════════
with t1:
    reps   = st.session_state.reps
    scores = [r["score"] for r in reps]

    k1,k2,k3,k4 = st.columns(4)
    with k1: st.metric("Total Reps", len(reps))
    with k2: st.metric("Avg Score", f"{np.mean(scores):.0f}" if scores else "—")
    with k3: st.metric("Best Rep",  max(scores) if scores else "—")
    with k4:
        dur = "—"
        if st.session_state.session_start:
            s = int((datetime.datetime.now()-st.session_state.session_start).total_seconds())
            dur = f"{s//60}:{s%60:02d}"
        st.metric("Duration", dur)

    st.markdown("<br/>", unsafe_allow_html=True)
    sc1, sc2 = st.columns(2, gap="large")

    with sc1:
        tl, slope = trend_calc(scores)
        ov        = overload_ok(reps)
        freq      = err_freq(reps)
        top_err   = list(freq.keys())[0] if freq else None

        _slope_span = ('<span style="font-size:11px;color:#3d7055;margin-left:8px;">' + f'{slope:+.1f} pts/rep</span>') if len(scores)>=4 else ''
        _overload_span = '<span style="color:#a8ff3e;font-family:Space Grotesk;font-weight:700;font-size:13px;">✓ Ready to increase load</span>' if ov else '<span style="color:#3d7055;font-size:12px;">Need 3 consecutive reps ≥80</span>'
        _top_err_span = (top_err.replace("_"," ").title() if top_err else "None — clean session")
        _status_html = (
            section_label("Session Status")
            + '<div style="display:flex;flex-direction:column;gap:16px;">'
            + '<div><p style="font-size:11px;color:#3d7055;margin-bottom:6px;">FORM TREND</p>'
            + trend_chip(tl) + _slope_span + '</div>'
            + '<div><p style="font-size:11px;color:#3d7055;margin-bottom:5px;">PROGRESSIVE OVERLOAD</p>'
            + _overload_span + '</div>'
            + '<div><p style="font-size:11px;color:#3d7055;margin-bottom:5px;">TOP ERROR</p>'
            + '<span style="font-size:13px;color:#ff8fa3;font-weight:600;">' + _top_err_span + '</span></div>'
            + '</div>'
        )
        st.markdown(card(_status_html), unsafe_allow_html=True)

    with sc2:
        fatigue = len(scores) >= 4 and tl == "declining"
        plateau = len(scores) >= 5 and (max(scores[-5:])-min(scores[-5:])) <= 5
        alerts  = []
        if fatigue:  alerts.append(("⚡","FATIGUE DETECTED","Form declining — rest or reduce load.","#ff4d6d","rgba(255,77,109,0.1)"))
        if ov:       alerts.append(("🚀","OVERLOAD READY","3 clean reps ≥80 — safe to add weight.","#a8ff3e","rgba(168,255,62,0.08)"))
        if plateau:  alerts.append(("📊","PLATEAU","Scores stable — vary technique or load.","#ffb800","rgba(255,184,0,0.08)"))
        if not alerts: alerts.append(("✓","ALL CLEAR","No alerts — keep training.","#00e6a0","rgba(0,230,160,0.06)"))

        alerts_html = "".join([alert_box(i,t,m,c,b) for i,t,m,c,b in alerts])
        st.markdown(card(f"{section_label('Alerts')}{alerts_html}"), unsafe_allow_html=True)

    # Bar chart
    if scores:
        st.markdown("<br/>", unsafe_allow_html=True)
        bars = ""
        for r in reps:
            s = r["score"]
            c = "#a8ff3e" if s>=75 else "#ffb800" if s>=50 else "#ff4d6d"
            bars += f"""<div style="display:flex;flex-direction:column;align-items:center;gap:3px;flex:1;min-width:20px;">
              <span style="font-size:9px;color:{c};font-weight:700;">{s}</span>
              <div style="width:100%;background:rgba(0,80,40,0.35);border-radius:5px;
                height:72px;display:flex;align-items:flex-end;overflow:hidden;">
                <div style="width:100%;height:{s}%;background:{c};
                  box-shadow:0 0 8px {c}66;border-radius:4px;"></div>
              </div>
              <span style="font-size:8px;color:#3d7055;">R{r['rep_number']}</span>
            </div>"""
        st.markdown(card(f"""
          {section_label("Rep Score History")}
          <div style="display:flex;align-items:flex-end;gap:5px;height:95px;overflow-x:auto;">{bars}</div>
        """), unsafe_allow_html=True)


# ════════════════════════════════════════════════
# TAB 2 — Rep Log
# ════════════════════════════════════════════════
with t2:
    reps = st.session_state.reps
    if not reps:
        st.markdown(card("""<div style="text-align:center;padding:36px 0;">
          <div style="font-size:36px;opacity:0.2;margin-bottom:10px;">🔁</div>
          <p style="font-family:Bebas Neue;font-size:18px;letter-spacing:0.08em;color:#3d7055;">
            No reps yet — simulate one or start a live session</p>
        </div>"""), unsafe_allow_html=True)
    else:
        for rep in reversed(reps):
            s   = rep["score"]
            det = [e for e,v in rep["errors"].items() if v]
            ph  = " ".join([pill(e,True) for e in det]) if det else \
                  '<span style="color:#00e6a0;font-size:11px;">✓ Clean rep</span>'
            with st.expander(f"Rep {rep['rep_number']}  —  {s}/100",
                             expanded=(rep["rep_number"]==len(reps))):
                lc, rc = st.columns([1,2])
                with lc:
                    _ring = score_ring_svg(s, 96)
                    st.markdown(
                        '<div style="display:flex;justify-content:center;margin:8px 0;">' + _ring + '</div>',
                        unsafe_allow_html=True)
                with rc:
                    st.markdown(f"""<div style="padding-top:6px;">
                      <div style="display:flex;flex-wrap:wrap;gap:5px;margin-bottom:10px;">{ph}</div>
                      {''.join([f'<p style="font-size:12px;color:#ff8fa3;margin:3px 0;">→ {fb}</p>' for fb in rep["feedback"]]) or '<p style="font-size:12px;color:#00e6a0;">No corrective feedback.</p>'}
                      <p style="font-size:10px;color:#3d7055;margin-top:10px;">{rep.get("timestamp","")[:19]}</p>
                    </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════
# TAB 3 — Analytics
# ════════════════════════════════════════════════
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
                pct  = int(cnt/mf*100)
                col  = "#ff4d6d" if pct>50 else "#ffb800" if pct>25 else "#00e6a0"
                fi  += f"""<div style="margin-bottom:13px;">
                  <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                    <span style="font-size:12px;color:rgba(232,245,238,0.85);">{err.replace('_',' ').title()}</span>
                    <span style="font-size:11px;color:{col};font-weight:700;">{cnt}/{len(reps)}</span>
                  </div>{mini_bar(pct,col)}</div>"""
            st.markdown(card(f"""{section_label("Error Frequency")}
              {fi if fi else '<p style="color:#3d7055;font-size:12px;">No errors detected.</p>'}
            """), unsafe_allow_html=True)

        with ra:
            bk = {"≥80 — Good":0,"60–79 — OK":0,"<60 — Poor":0}
            bc = {"≥80 — Good":"#a8ff3e","60–79 — OK":"#ffb800","<60 — Poor":"#ff4d6d"}
            for s in scores:
                if s>=80: bk["≥80 — Good"]+=1
                elif s>=60: bk["60–79 — OK"]+=1
                else: bk["<60 — Poor"]+=1
            di = ""
            for lb,cnt in bk.items():
                pct = int(cnt/len(scores)*100) if scores else 0
                di += f"""<div style="margin-bottom:13px;">
                  <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                    <span style="font-size:12px;color:rgba(232,245,238,0.85);">{lb}</span>
                    <span style="font-size:11px;color:{bc[lb]};font-weight:700;">{cnt} ({pct}%)</span>
                  </div>{mini_bar(pct,bc[lb])}</div>"""
            st.markdown(card(f"{section_label('Score Distribution')}{di}"), unsafe_allow_html=True)

        if freq:
            top = list(freq.keys())[0]
            drills = {
                "knees_inward":"3×12 banded squats — cue knee tracking outward.",
                "knees_forward":"Wall squat drill — toes 2cm from wall.",
                "rounded_back":"Paused Romanian deadlift — reinforce neutral spine.",
                "shallow_squat":"Box squat to low box — build depth confidence.",
                "elbow_error":"Z-press (seated) — isolate vertical pressing path.",
                "knees_error":"Tempo OHP 3-1-1 — slow eccentric for control.",
                "spine_error":"Dead-stop OHP from rack pins at forehead height.",
                "lumbar_error":"Chest-supported row — remove spinal loading.",
                "torso_angle_error":"Paused rows — hold torso angle at contraction.",
            }
            st.markdown("<br/>", unsafe_allow_html=True)
            st.markdown(card(f"""
              {section_label("Top Coaching Recommendation")}
              <div style="display:flex;gap:14px;align-items:flex-start;">
                <span style="font-size:28px;flex-shrink:0;">🎯</span>
                <div>
                  <p style="font-family:Bebas Neue;font-size:18px;letter-spacing:0.06em;
                    color:#ff8fa3;margin-bottom:6px;">{top.replace("_"," ").upper()}</p>
                  <p style="font-size:13px;color:rgba(232,245,238,0.82);line-height:1.6;">
                    {drills.get(top,"Review form with a qualified coach.")}
                  </p>
                </div>
              </div>
            """), unsafe_allow_html=True)


# ════════════════════════════════════════════════
# TAB 4 — Setup Guide
# ════════════════════════════════════════════════
with t4:
    steps = [
        ("01","Install Dependencies",
         "python -m venv venv && source venv/bin/activate\npip install mediapipe opencv-python numpy scipy scikit-learn streamlit joblib",
         "Python 3.10+ · CPU-only · GPU optional"),
        ("02","Project Structure",
         "gym_aq/\n├── app.py          ← Streamlit dashboard\n├── main.py         ← Live webcam CLI\n├── train.py        ← Model training\n├── pipeline/       ← Data pipeline modules\n├── modules/        ← Core AI modules\n├── models/         ← Trained .pkl files\n└── data/           ← Fitness-AQA dataset",
         "Each module is independently importable"),
        ("03","Train All Models",
         "python train.py --exercise all\n# Or individually:\npython train.py --exercise Squat\npython train.py --exercise OHP\npython train.py --exercise BarbellRow",
         "Models saved to models/ — ~30–120 min per exercise on CPU"),
        ("04","Launch Dashboard",
         "streamlit run app.py",
         "Opens at http://localhost:8501 · Demo mode works without webcam"),
        ("05","Live Webcam Mode",
         "python main.py --exercise BackSquat\npython main.py --exercise OverheadPress\npython main.py --exercise BarbellRow\n# Press Q to quit",
         "Auto-resolves model path from models/ folder"),
    ]
    for num, title, code, note in steps:
        st.markdown(card(f"""
          <div style="display:flex;gap:16px;align-items:flex-start;">
            <div style="flex-shrink:0;width:34px;height:34px;border-radius:9px;
              background:linear-gradient(135deg,rgba(168,255,62,0.15),rgba(0,230,160,0.1));
              border:1px solid rgba(0,230,160,0.25);display:flex;align-items:center;
              justify-content:center;font-family:Bebas Neue,sans-serif;
              font-size:14px;letter-spacing:0.06em;color:#00e6a0;">{num}</div>
            <div style="flex:1;min-width:0;">
              <p style="font-family:Bebas Neue,sans-serif;font-size:16px;
                letter-spacing:0.06em;color:#e8f5ee;margin-bottom:8px;">{title}</p>
              <pre style="background:rgba(0,10,8,0.7);border:1px solid rgba(0,230,160,0.1);
                border-radius:9px;padding:11px;font-size:11px;color:#7bbf9a;
                overflow-x:auto;margin-bottom:7px;font-family:JetBrains Mono,monospace;
                line-height:1.65;">{code}</pre>
              <p style="font-size:11px;color:#3d7055;letter-spacing:0.03em;">{note}</p>
            </div>
          </div>
        ""","margin-bottom:10px;"), unsafe_allow_html=True)


# ── Auto-refresh ──────────────────────────────────────────────────────────────
if st.session_state.running:
    time.sleep(0.08)
    st.rerun()