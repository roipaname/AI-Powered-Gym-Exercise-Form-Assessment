

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS
# ══════════════════════════════════════════════════════════════════════════════

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Bebas+Neue&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --lime:    #a8ff3e;
  --teal:    #00e6a0;
  --cyan:    #00c8e0;
  --amber:   #ffb800;
  --red:     #ff4d6d;
  --ink:     #03080a;
  --rim:     rgba(0,230,160,0.14);
  --rim2:    rgba(0,230,160,0.28);
  --tx:      #e8f5ee;
  --tx2:     #7bbf9a;
  --tx3:     #3d7055;
  --glass:   linear-gradient(135deg,rgba(0,230,160,0.07),rgba(0,200,224,0.04));
  --grad:    linear-gradient(135deg,#a8ff3e 0%,#00e6a0 50%,#00c8e0 100%);
}

/* ── Reset ── */
*,*::before,*::after{box-sizing:border-box;}
html,body,[data-testid="stAppViewContainer"],[data-testid="stApp"],.main,.block-container{
  background:transparent!important;
}

/* ── Background ── */
[data-testid="stApp"]::before{
  content:'';position:fixed;inset:0;z-index:-2;
  background:
    radial-gradient(ellipse 70% 60% at 15% 10%,rgba(0,180,120,0.12) 0%,transparent 65%),
    radial-gradient(ellipse 50% 70% at 85% 85%,rgba(0,150,200,0.10) 0%,transparent 60%),
    radial-gradient(ellipse 80% 80% at 50% 50%,#03080a 0%,#010608 100%);
}
[data-testid="stApp"]::after{
  content:'';position:fixed;inset:0;z-index:-1;pointer-events:none;
  background-image:radial-gradient(circle,rgba(0,230,160,0.05) 1px,transparent 1px);
  background-size:32px 32px;
}

/* ── Typography ── */
body,p,span,div,label{font-family:'Space Grotesk',sans-serif!important;color:var(--tx)!important;}
h1,h2,h3,h4{font-family:'Bebas Neue',sans-serif!important;letter-spacing:0.06em;color:var(--tx)!important;}
code,pre{font-family:'JetBrains Mono',monospace!important;}

/* ── Sidebar ── */
[data-testid="stSidebar"]{
  background:rgba(3,10,12,0.92)!important;
  border-right:1px solid var(--rim)!important;
  backdrop-filter:blur(28px)!important;
}
[data-testid="stSidebar"] *{color:var(--tx)!important;font-family:'Space Grotesk',sans-serif!important;}

/* ── Inputs ── */
.stSelectbox>div>div{background:rgba(0,230,160,0.06)!important;border:1px solid var(--rim)!important;border-radius:10px!important;}
.stSelectbox [data-baseweb="select"]>div{background:transparent!important;border:none!important;}
.stTextInput>div>div>input{background:rgba(0,230,160,0.05)!important;border:1px solid var(--rim)!important;border-radius:10px!important;color:var(--tx)!important;}

/* ── Buttons ── */
.stButton>button{
  background:var(--glass)!important;
  border:1px solid var(--rim)!important;border-radius:10px!important;
  color:var(--teal)!important;font-family:'Space Grotesk',sans-serif!important;
  font-weight:600!important;font-size:13px!important;letter-spacing:0.04em!important;
  padding:9px 18px!important;transition:all 0.18s ease!important;width:100%!important;
}
.stButton>button:hover{
  background:rgba(0,230,160,0.14)!important;border-color:var(--rim2)!important;
  box-shadow:0 0 20px rgba(0,230,160,0.18)!important;color:#fff!important;
  transform:translateY(-1px);
}
.stDownloadButton>button{
  background:rgba(0,230,160,0.1)!important;border:1px solid var(--rim2)!important;
  border-radius:10px!important;color:var(--lime)!important;
  font-family:'Space Grotesk',sans-serif!important;font-weight:600!important;width:100%!important;
}

/* ── Metrics ── */
[data-testid="metric-container"]{
  background:var(--glass)!important;border:1px solid var(--rim)!important;
  border-radius:14px!important;padding:18px!important;backdrop-filter:blur(12px)!important;
  transition:border-color 0.2s;
}
[data-testid="metric-container"]:hover{border-color:var(--rim2)!important;}
[data-testid="stMetricValue"]{
  font-family:'Bebas Neue',sans-serif!important;font-size:2.4rem!important;
  background:var(--grad);-webkit-background-clip:text;
  -webkit-text-fill-color:transparent;background-clip:text;
}
[data-testid="stMetricLabel"]{
  font-size:11px!important;text-transform:uppercase!important;
  letter-spacing:0.12em!important;color:var(--tx3)!important;font-weight:600!important;
}
[data-testid="stMetricDelta"] svg{display:none;}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"]{
  background:rgba(0,230,160,0.04)!important;border:1px solid var(--rim)!important;
  border-radius:12px!important;padding:4px!important;gap:3px!important;
}
.stTabs [data-baseweb="tab"]{
  background:transparent!important;color:var(--tx3)!important;
  font-family:'Space Grotesk',sans-serif!important;font-weight:600!important;
  font-size:12px!important;letter-spacing:0.06em!important;border-radius:9px!important;
  padding:7px 18px!important;border:none!important;text-transform:uppercase!important;
}
.stTabs [aria-selected="true"]{
  background:rgba(0,230,160,0.12)!important;color:var(--teal)!important;
  box-shadow:0 0 14px rgba(0,230,160,0.15)!important;
}
.stTabs [data-baseweb="tab-border"]{display:none!important;}

/* ── Progress ── */
.stProgress>div>div>div{
  background:var(--grad)!important;border-radius:99px!important;
  box-shadow:0 0 10px rgba(0,230,160,0.3)!important;
}
.stProgress>div>div{background:rgba(0,230,160,0.08)!important;border-radius:99px!important;}

/* ── Expander ── */
.streamlit-expanderHeader{
  background:rgba(0,230,160,0.05)!important;border:1px solid var(--rim)!important;
  border-radius:10px!important;font-family:'Space Grotesk',sans-serif!important;
}

/* ── WebRTC video — full width, no shrink ── */
.webrtc-wrap{border-radius:16px;overflow:hidden;width:100%;}
.webrtc-wrap>div{width:100%!important;}
.webrtc-wrap video{width:100%!important;height:auto!important;
  border-radius:16px!important;display:block!important;
  border:1px solid rgba(0,230,160,0.25)!important;}
.webrtc-wrap button[kind="primary"]{
  background:linear-gradient(135deg,rgba(0,230,160,0.22),rgba(0,200,224,0.15))!important;
  border:1px solid rgba(0,230,160,0.4)!important;color:#a8ff3e!important;
  font-family:'Space Grotesk',sans-serif!important;font-weight:700!important;
  letter-spacing:0.06em!important;border-radius:10px!important;
  padding:10px 28px!important;font-size:14px!important;
}
iframe[title="streamlit_webrtc.frontend"]{width:100%!important;min-height:420px!important;}

/* ── Misc ── */
hr{border-color:var(--rim)!important;margin:14px 0!important;}
.stAlert{background:rgba(0,230,160,0.06)!important;border:1px solid var(--rim)!important;border-radius:12px!important;}
::-webkit-scrollbar{width:5px;height:5px;}
::-webkit-scrollbar-track{background:transparent;}
::-webkit-scrollbar-thumb{background:var(--tx3);border-radius:99px;}
#MainMenu,footer,header,[data-testid="stToolbar"]{visibility:hidden;}
.block-container{padding-top:1.8rem!important;}
@keyframes pulse{0%,100%{opacity:1;}50%{opacity:0.35;}}
/* Hide Streamlit exception/error UI — errors go to terminal only */
[data-testid="stException"],[data-testid="stException"] *{display:none!important;}
.element-container:has([data-testid="stException"]){display:none!important;}
/* Hide the webrtc warning banner */
[data-testid="stAlert"] p:contains("Connection is taking"){
  opacity:0;pointer-events:none;
}
</style>
"""

# ══════════════════════════════════════════════════════════════════════════════
# HTML COMPONENT BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def card(html: str, extra: str = "") -> str:
    return (
        '<div style="background:linear-gradient(135deg,rgba(0,230,160,0.07),'
        'rgba(0,200,224,0.04));border:1px solid rgba(0,230,160,0.14);'
        'border-radius:16px;padding:22px;backdrop-filter:blur(18px);'
        '-webkit-backdrop-filter:blur(18px);' + extra + '">'
        + html + '</div>'
    )


def section_label(text: str) -> str:
    return (
        '<p style="font-family:Space Grotesk,sans-serif;font-size:10px;font-weight:700;'
        'letter-spacing:0.14em;text-transform:uppercase;color:#3d7055;margin-bottom:10px;">'
        + text + '</p>'
    )


def score_ring_svg(score, size: int = 128) -> str:
    pct   = max(0, min(100, score if isinstance(score, int) else 0))
    r     = 44
    circ  = 2 * 3.14159 * r
    dash  = circ * pct / 100
    offset = circ / 4
    if pct >= 75:   col, glow = "#a8ff3e", "rgba(168,255,62,0.5)"
    elif pct >= 50: col, glow = "#ffb800", "rgba(255,184,0,0.5)"
    else:           col, glow = "#ff4d6d", "rgba(255,77,109,0.5)"
    label = str(score) if isinstance(score, int) else "--"
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 100 100">'
        f'<circle cx="50" cy="50" r="{r}" fill="none" stroke="rgba(0,80,50,0.5)" stroke-width="7"/>'
        f'<circle cx="50" cy="50" r="{r}" fill="none" stroke="{col}" stroke-width="7"'
        f' stroke-dasharray="{dash:.1f} {circ:.1f}" stroke-dashoffset="{offset:.1f}"'
        f' stroke-linecap="round" style="filter:drop-shadow(0 0 7px {glow});"/>'
        f'<text x="50" y="46" text-anchor="middle" font-size="21" font-weight="700"'
        f' font-family="Bebas Neue,sans-serif" fill="{col}" letter-spacing="1">{label}</text>'
        f'<text x="50" y="60" text-anchor="middle" font-size="8"'
        f' font-family="Space Grotesk,sans-serif" fill="rgba(123,191,154,0.6)">/100</text>'
        f'</svg>'
    )


def pill(label: str, detected: bool) -> str:
    if detected:
        bg, bd, col, ic = "rgba(255,77,109,0.12)", "rgba(255,77,109,0.4)", "#ff8fa3", "✕"
    else:
        bg, bd, col, ic = "rgba(0,230,160,0.08)", "rgba(0,230,160,0.25)", "#00e6a0", "✓"
    return (
        f'<span style="background:{bg};border:1px solid {bd};color:{col};'
        'font-family:Space Grotesk,sans-serif;font-size:11px;font-weight:600;'
        'padding:3px 11px;border-radius:99px;display:inline-flex;align-items:center;'
        f'gap:5px;letter-spacing:0.03em;white-space:nowrap;">{ic} {label.replace("_"," ")}</span>'
    )


def trend_chip(label: str) -> str:
    m = {
        "improving":         ("↗", "#a8ff3e", "rgba(168,255,62,0.1)",  "rgba(168,255,62,0.28)"),
        "stable":            ("→", "#00e6a0", "rgba(0,230,160,0.08)",  "rgba(0,230,160,0.22)"),
        "declining":         ("↘", "#ff4d6d", "rgba(255,77,109,0.1)",  "rgba(255,77,109,0.3)"),
        "insufficient data": ("·", "#3d7055", "rgba(61,112,85,0.1)",   "rgba(61,112,85,0.2)"),
    }
    ic, col, bg, bd = m.get(label, m["insufficient data"])
    return (
        f'<span style="background:{bg};border:1px solid {bd};color:{col};'
        'font-family:Bebas Neue,sans-serif;font-size:15px;letter-spacing:0.08em;'
        f'padding:4px 14px;border-radius:99px;display:inline-flex;align-items:center;gap:7px;">'
        f'{ic} {label.upper()}</span>'
    )


def mini_bar(pct: float, color: str = "#00e6a0") -> str:
    return (
        '<div style="width:100%;height:5px;background:rgba(0,100,60,0.3);'
        'border-radius:99px;overflow:hidden;margin-top:5px;">'
        f'<div style="width:{min(100,int(pct))}%;height:100%;background:{color};'
        f'border-radius:99px;box-shadow:0 0 8px {color}44;"></div></div>'
    )


def alert_box(icon: str, title: str, msg: str, col: str, bg: str) -> str:
    return (
        f'<div style="background:{bg};border:1px solid {col}55;border-radius:12px;'
        f'padding:12px 14px;margin-bottom:8px;">'
        f'<p style="font-family:Bebas Neue,sans-serif;font-size:15px;letter-spacing:0.06em;'
        f'color:{col};margin-bottom:3px;">{icon} {title}</p>'
        f'<p style="font-size:12px;color:rgba(232,245,238,0.75);line-height:1.5;">{msg}</p>'
        f'</div>'
    )


def status_dot(live: bool) -> str:
    col = "#a8ff3e" if live else "#3d7055"
    anim = "animation:pulse 1.2s infinite;" if live else ""
    label = "LIVE" if live else "IDLE"
    return (
        f'<div style="display:flex;align-items:center;gap:9px;padding:10px 12px;'
        'background:rgba(0,230,160,0.05);border:1px solid rgba(0,230,160,0.12);border-radius:10px;">'
        f'<span style="width:7px;height:7px;border-radius:50%;background:{col};'
        f'box-shadow:{"0 0 8px "+col if live else "none"};{anim}"></span>'
        f'<span style="font-family:Bebas Neue,sans-serif;font-size:14px;letter-spacing:0.12em;'
        f'color:{col};">{label}</span>'
        f'</div>'
        '<style>@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.3}}</style>'
    )