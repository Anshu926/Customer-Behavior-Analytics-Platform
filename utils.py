"""
Utility functions for model training, loading, data processing, and enterprise styling.
"""
import streamlit as st
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# CONSTANTS & FEATURE MAPPINGS
# ============================================================================
DATA_FILE = "clean_customer_data.csv"
PREDICTOR_MODEL = "predictor.pkl"
CLUSTER_MODEL = "cluster.pkl"
SCALER_MODEL = "scaler.pkl"

MEMBERSHIP_MAPPING = {"Bronze": 1, "Silver": 2, "Gold": 3}
MEMBERSHIP_REVERSE = {1: "Bronze", 2: "Silver", 3: "Gold"}
GENDER_MAPPING = {"Female": 0, "Male": 1}
GENDER_REVERSE = {0: "Female", 1: "Male"}
SATISFACTION_MAPPING = {0: "Unsatisfied", 1: "Neutral", 2: "Satisfied"}

# ============================================================================
# FONT AWESOME CDN & GLOBAL DESIGN SYSTEM
# ============================================================================
FA_CDN = """
<link rel="stylesheet"
      href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css"
      integrity="sha512-DTOQO9RWCH3ppGqcWaEA1BIZOC6xxalwEsw9c2QQeAIftl+Vegovlnee1c9QX4TctnWMn13TZye+giMm8e2LwA=="
      crossorigin="anonymous" referrerpolicy="no-referrer">
"""

GLOBAL_CSS = """
<style>
/* ── Design tokens ─────────────────────────────────────────────────────────── */
:root {
    --bg-base:      #0A0E17;
    --bg-card:      #111827;
    --bg-elevated:  #1A2234;
    --border:       #1F2D45;
    --border-soft:  #243044;
    --accent:       #6366F1;
    --accent-glow:  rgba(99,102,241,0.22);
    --green:        #10B981;
    --amber:        #F59E0B;
    --red:          #EF4444;
    --text-primary: #F1F5F9;
    --text-soft:    #94A3B8;
    --text-muted:   #64748B;
}

/* ── Base ──────────────────────────────────────────────────────────────────── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stApp"] {
    background-color: var(--bg-base) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}
[data-testid="stMain"], .main, .block-container {
    background-color: var(--bg-base) !important;
    padding-top: 1rem !important;
    max-width: 1200px;
}

/* ── Typography ────────────────────────────────────────────────────────────── */
h1 {
    color: var(--text-primary) !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.04em !important;
    line-height: 1.2 !important;
    margin: 0 0 0.4rem 0 !important;
}
h2 {
    color: var(--text-primary) !important;
    font-size: 1.3rem !important;
    font-weight: 600 !important;
    letter-spacing: -0.02em !important;
    margin: 1.5rem 0 0.75rem 0 !important;
}
h3 {
    color: var(--text-primary) !important;
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    margin: 1rem 0 0.5rem 0 !important;
}
p:not(button p):not(button *):not([data-testid*="Button"] *):not([data-testid="stButton"] *) {
    color: var(--text-soft) !important;
    font-size: 0.9rem !important;
    line-height: 1.65 !important;
}
label { color: var(--text-soft) !important; font-size: 0.85rem !important; font-weight: 500 !important; }
hr  { border: none !important; border-top: 1px solid var(--border) !important; margin: 1.5rem 0 !important; }

/* ── Remove Top Header, Deploy Button & 3-Dots Options Menu ───────────────── */
#MainMenu,
[data-testid="stMainMenu"],
[data-testid="stDeployButton"],
.stDeployButton,
header[data-testid="stHeader"],
.stAppHeader,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
[data-testid="stActionMenu"],
footer,
button[aria-label="Manage app"],
button[title="View app in Streamlit Community Cloud"],
button[data-testid="baseButton-headerNoPadding"],
div[data-testid="stToolbar"] {
    display: none !important;
    visibility: hidden !important;
    height: 0 !important;
    width: 0 !important;
    opacity: 0 !important;
    pointer-events: none !important;
}

/* ── Sidebar shell & Permanent Open ────────────────────────────────────────── */

[data-testid="stSidebarNav"],
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"],
[data-testid="stSidebarHeader"] button,
button[aria-label="Close sidebar"],
button[aria-label="Open sidebar"],
button[data-testid="stSidebarCollapseButton"],
div[data-testid="stSidebarCollapseButton"] {
    display: none !important;
    visibility: hidden !important;
    pointer-events: none !important;
    width: 0 !important;
    height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
}

section[data-testid="stSidebar"] {
    background: #080D16 !important;
    border-right: 1px solid rgba(99,102,241,0.12) !important;
    min-width: 270px !important;
    max-width: 285px !important;
    display: block !important;
    transform: none !important;
    margin-left: 0 !important;
}
section[data-testid="stSidebar"] > div { padding: 0 !important; }


/* ── Sidebar nav links ─────────────────────────────────────────────────────── */
.sb-nav-link {
    display: flex !important;
    align-items: center !important;
    gap: 0.75rem !important;
    padding: 0.65rem 1rem !important;
    margin: 0.15rem 0.6rem !important;
    border-radius: 10px !important;
    color: #94A3B8 !important;
    text-decoration: none !important;
    font-size: 0.86rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.01em !important;
    transition: all 0.2s ease !important;
    border-left: 3px solid transparent !important;
}
.sb-nav-link:hover {
    background: rgba(99,102,241,0.1) !important;
    color: #F1F5F9 !important;
    border-left-color: rgba(99,102,241,0.4) !important;
    text-decoration: none !important;
    transform: translateX(2px);
}
.sb-nav-link.active {
    background: linear-gradient(90deg, rgba(99,102,241,0.2) 0%, rgba(99,102,241,0.05) 100%) !important;
    color: #A5B4FC !important;
    border-left-color: #6366F1 !important;
    font-weight: 600 !important;
}
.sb-nav-link .nav-icon {
    width: 20px;
    text-align: center;
    font-size: 0.88rem;
    flex-shrink: 0;
    color: #64748B;
    transition: color 0.2s ease;
}
.sb-nav-link:hover .nav-icon { color: #A5B4FC; }
.sb-nav-link.active .nav-icon { color: #6366F1; }
.sb-nav-link .nav-label { flex: 1; }
.sb-nav-link .nav-badge {
    background: rgba(99,102,241,0.25);
    color: #A5B4FC;
    font-size: 0.6rem;
    font-weight: 700;
    padding: 0.12rem 0.4rem;
    border-radius: 4px;
    letter-spacing: 0.06em;
}

/* ── Metric widget ─────────────────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 1.1rem 1.25rem !important;
}
[data-testid="stMetricLabel"] p {
    color: var(--text-muted) !important; font-size: 0.72rem !important;
    font-weight: 600 !important; text-transform: uppercase !important; letter-spacing: 0.08em !important;
}
[data-testid="stMetricValue"] { color: var(--text-primary) !important; font-size: 1.55rem !important; font-weight: 700 !important; }
[data-testid="stMetricDelta"] { font-size: 0.75rem !important; }

/* ── Ultra-Forceful Button Text & Style ────────────────────────────────────── */
button,
button *,
button p,
button span,
button div,
[data-testid*="Button"],
[data-testid*="Button"] *,
[data-testid*="Button"] p,
[data-testid*="Button"] span,
[data-testid*="Button"] div,
.stButton,
.stButton *,
.stButton button,
.stButton button *,
.stButton button p,
.stButton button span,
.stButton button div,
[data-testid="baseButton-secondary"],
[data-testid="baseButton-secondary"] *,
[data-testid="baseButton-secondary"] p,
[data-testid="baseButton-primary"],
[data-testid="baseButton-primary"] *,
[data-testid="baseButton-primary"] p,
[data-testid="stBaseButton-secondary"],
[data-testid="stBaseButton-secondary"] *,
[data-testid="stBaseButton-secondary"] p,
[data-testid="stBaseButton-primary"],
[data-testid="stBaseButton-primary"] *,
[data-testid="stBaseButton-primary"] p,
button[kind="secondary"],
button[kind="secondary"] *,
button[kind="secondary"] p,
button[kind="primary"],
button[kind="primary"] *,
button[kind="primary"] p {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.02em !important;
}

div[data-testid="stButton"] > button,
div[data-testid="stButton"] button,
button[data-testid*="Button"],
button[kind="secondary"],
button[kind="primary"] {
    background: linear-gradient(135deg, #6366F1 0%, #4F46E5 100%) !important;
    border: 1px solid rgba(255, 255, 255, 0.25) !important;
    border-radius: 10px !important;
    padding: 0.65rem 1.4rem !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 14px rgba(99,102,241,0.4) !important;
    min-height: 44px !important;
}

/* ── Dataset / Kaggle Action Buttons (Forced White Text) ───────────────────── */
.kaggle-btn,
.kaggle-btn *,
.kaggle-btn span,
.kaggle-btn i,
.kaggle-btn svg,
a.kaggle-btn,
a.kaggle-btn:hover,
a.kaggle-btn:visited,
a.kaggle-btn:active,
a.kaggle-btn * {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    text-decoration: none !important;
}
.kaggle-btn {
    display: inline-flex !important;
    align-items: center !important;
    gap: 0.5rem !important;
    background: linear-gradient(135deg, #4F46E5 0%, #6366F1 100%) !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    border-radius: 8px !important;
    padding: 0.6rem 1.25rem !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 14px rgba(99, 102, 241, 0.4) !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease !important;
}
.kaggle-btn:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 18px rgba(99, 102, 241, 0.55) !important;
}

div[data-testid="stButton"] > button:hover,
div[data-testid="stButton"] button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(99,102,241,0.6) !important;
    filter: brightness(1.12) !important;
    border-color: rgba(255, 255, 255, 0.5) !important;
}

div[data-testid="stButton"] > button:active,
div[data-testid="stButton"] button:active {
    transform: translateY(0) !important;
    filter: brightness(0.95) !important;
}

/* ── Inputs ────────────────────────────────────────────────────────────────── */

[data-testid="stSelectbox"] > div > div,
[data-baseweb="select"] > div {
    background: var(--bg-card) !important; border: 1px solid var(--border) !important;
    border-radius: 8px !important; color: var(--text-primary) !important;
}

/* ── Alert boxes ───────────────────────────────────────────────────────────── */
[data-testid="stInfo"]    { background: rgba(99,102,241,0.08)  !important; border: 1px solid rgba(99,102,241,0.28)  !important; border-radius: 10px !important; }
[data-testid="stWarning"] { background: rgba(245,158,11,0.08)  !important; border: 1px solid rgba(245,158,11,0.28)  !important; border-radius: 10px !important; }
[data-testid="stError"]   { background: rgba(239,68,68,0.08)   !important; border: 1px solid rgba(239,68,68,0.28)   !important; border-radius: 10px !important; }
[data-testid="stSuccess"] { background: rgba(16,185,129,0.08)  !important; border: 1px solid rgba(16,185,129,0.28)  !important; border-radius: 10px !important; }

/* ── DataFrame ─────────────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] { border-radius: 12px !important; overflow: hidden !important; border: 1px solid var(--border) !important; }

/* ── Animations ────────────────────────────────────────────────────────────── */
@keyframes fadeInUp  { from { opacity:0; transform:translateY(14px); } to { opacity:1; transform:translateY(0); } }
@keyframes fadeIn    { from { opacity:0; } to { opacity:1; } }
@keyframes glowPulse { 0%,100% { box-shadow:0 0 14px var(--accent-glow); } 50% { box-shadow:0 0 32px var(--accent-glow); } }

/* ═══════════════════════════════════════════════════════════════
   REUSABLE COMPONENT CLASSES
   ═══════════════════════════════════════════════════════════════ */

/* Glass metric card */
.pro-card {
    background: linear-gradient(135deg, #111827 0%, #1A2234 100%);
    border: 1px solid var(--border-soft);
    border-radius: 14px;
    padding: 1.25rem 1.4rem;
    text-align: center;
    transition: transform 0.22s ease, box-shadow 0.22s ease, border-color 0.22s ease;
    animation: fadeInUp 0.35s ease both;
}
.pro-card:hover { transform: translateY(-3px); box-shadow: 0 12px 36px rgba(0,0,0,0.45); border-color: rgba(99,102,241,0.35); }
.pro-card .card-fa-icon { font-size: 1.2rem; color: var(--accent); margin-bottom: 0.6rem; display: block; }
.pro-card .card-label { color: var(--text-muted); font-size: 0.68rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 0.5rem; }
.pro-card .card-value { color: var(--text-primary); font-size: 1.85rem; font-weight: 700; letter-spacing: -0.03em; line-height: 1.1; margin-bottom: 0.4rem; }
.pro-card .card-delta { font-size: 0.75rem; font-weight: 500; }
.card-delta.positive { color: #34D399; }
.card-delta.neutral  { color: #FCD34D; }
.card-delta.info     { color: #60A5FA; }
.card-delta.muted    { color: var(--text-muted); }

/* Hero banner */
.hero-banner {
    background: linear-gradient(135deg, #0F172A 0%, #1E1B4B 55%, #0F172A 100%);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 18px;
    padding: 2.2rem 2.6rem;
    margin-bottom: 1.75rem;
    position: relative; overflow: hidden;
    animation: fadeIn 0.5s ease both;
}
.hero-banner::before {
    content: ''; position: absolute; top: -60%; right: -8%;
    width: 340px; height: 340px;
    background: radial-gradient(circle, rgba(99,102,241,0.13) 0%, transparent 68%);
    pointer-events: none;
}
.hero-banner::after {
    content: ''; position: absolute; bottom: -40%; left: -5%;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(16,185,129,0.07) 0%, transparent 68%);
    pointer-events: none;
}
.hero-tag {
    display: inline-flex; align-items: center; gap: 0.45rem;
    background: rgba(99,102,241,0.14);
    border: 1px solid rgba(99,102,241,0.32);
    color: #A5B4FC;
    font-size: 0.7rem; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase;
    padding: 0.22rem 0.7rem; border-radius: 50px; margin-bottom: 0.85rem;
}
.hero-banner h1 {
    font-size: 2.1rem !important;
    background: linear-gradient(135deg, #F1F5F9 30%, #A5B4FC 100%) !important;
    -webkit-background-clip: text !important; -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    margin: 0 0 0.6rem 0 !important; letter-spacing: -0.05em !important;
}
.hero-sub { color: var(--text-muted); font-size: 0.92rem; line-height: 1.65; max-width: 580px; }

/* Section header */
.section-header { display: flex; align-items: center; gap: 0.6rem; margin: 1.6rem 0 1rem 0; }
.section-accent { width: 4px; height: 20px; background: linear-gradient(180deg, #6366F1 0%, #4F46E5 100%); border-radius: 2px; flex-shrink: 0; }

/* Feature card */
.feature-card {
    background: var(--bg-card); border: 1px solid var(--border); border-radius: 14px;
    padding: 1.4rem 1.5rem; height: 100%;
    transition: all 0.22s ease; animation: fadeInUp 0.4s ease both;
}
.feature-card:hover { border-color: rgba(99,102,241,0.4); transform: translateY(-2px); box-shadow: 0 8px 30px rgba(0,0,0,0.35); }
.feature-card .fc-icon-wrap {
    width: 40px; height: 40px;
    background: rgba(99,102,241,0.1);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    margin-bottom: 0.85rem;
    font-size: 0.95rem; color: var(--accent);
}
.feature-card .fc-title { color: var(--text-primary); font-size: 0.95rem; font-weight: 600; margin-bottom: 0.5rem; }
.feature-card .fc-desc  { color: var(--text-muted); font-size: 0.82rem; line-height: 1.6; }

/* Insight panel */
.insight-panel {
    background: var(--bg-card); border: 1px solid var(--border-soft); border-radius: 12px;
    padding: 1.3rem 1.5rem; margin-bottom: 1rem;
    animation: fadeInUp 0.4s ease both; transition: border-color 0.22s ease;
}
.insight-panel:hover { border-color: rgba(99,102,241,0.38); }
.insight-panel .ip-num { color: var(--accent); font-size: 0.68rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 0.35rem; }
.insight-panel h4 { color: var(--text-primary) !important; font-size: 0.97rem !important; font-weight: 600 !important; margin: 0 0 0.55rem 0 !important; }

/* Recommendation card */
.rec-card {
    background: var(--bg-card); border: 1px solid var(--border-soft);
    border-top: 3px solid var(--accent); border-radius: 12px;
    padding: 1.2rem 1.4rem; margin-bottom: 0.85rem;
    animation: fadeInUp 0.4s ease both;
}
.rec-card .rc-title { color: var(--text-primary); font-size: 0.95rem; font-weight: 600; margin-bottom: 0.45rem; }
.rec-card .rc-desc  { color: var(--text-soft); font-size: 0.82rem; line-height: 1.55; margin-bottom: 0.6rem; }
.rc-meta { display: flex; gap: 0.4rem; flex-wrap: wrap; align-items: center; }

/* Segment card */
.seg-card {
    background: var(--bg-card); border: 1px solid var(--border-soft); border-radius: 14px;
    padding: 1.3rem 1.5rem; margin-bottom: 1rem;
    animation: fadeInUp 0.4s ease both; transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.seg-card:hover { transform: translateY(-2px); box-shadow: 0 8px 28px rgba(0,0,0,0.35); }
.seg-card .sc-header { display: flex; align-items: center; gap: 0.65rem; margin-bottom: 0.9rem; padding-bottom: 0.7rem; border-bottom: 1px solid var(--border); }
.seg-card .sc-stripe  { width: 4px; height: 28px; border-radius: 2px; flex-shrink: 0; }
.seg-card .sc-icon-wrap { width: 32px; height: 32px; border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 0.8rem; flex-shrink: 0; }
.seg-card .sc-title  { color: var(--text-primary); font-size: 0.97rem; font-weight: 600; }
.seg-card .sc-stats   { display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem 1rem; }
.seg-card .sc-stat-label { color: var(--text-muted); font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.1rem; }
.seg-card .sc-stat-value { color: var(--text-primary); font-size: 0.9rem; font-weight: 600; }

/* Badges */
.badge { display: inline-block; padding: 0.18rem 0.55rem; border-radius: 50px; font-size: 0.68rem; font-weight: 600; letter-spacing: 0.04em; }
.badge-green  { background: rgba(16,185,129,0.14);  color: #34D399; border: 1px solid rgba(16,185,129,0.28); }
.badge-amber  { background: rgba(245,158,11,0.14);  color: #FCD34D; border: 1px solid rgba(245,158,11,0.28); }
.badge-red    { background: rgba(239,68,68,0.14);   color: #FCA5A5; border: 1px solid rgba(239,68,68,0.28); }
.badge-indigo { background: rgba(99,102,241,0.14);  color: #A5B4FC; border: 1px solid rgba(99,102,241,0.28); }
.badge-gray   { background: rgba(100,116,139,0.14); color: #94A3B8; border: 1px solid rgba(100,116,139,0.28); }

/* Result display */
.result-card { border-radius: 14px; padding: 1.6rem; text-align: center; animation: fadeInUp 0.4s ease both; }
.result-card .rc-label  { font-size: 0.7rem; font-weight: 600; letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 0.5rem; opacity: 0.8; }
.result-card .rc-value  { font-size: 1.6rem; font-weight: 700; letter-spacing: -0.02em; }
</style>
"""


def apply_global_styles():
    """Inject Font Awesome CDN + design-system CSS."""
    st.markdown(FA_CDN + GLOBAL_CSS, unsafe_allow_html=True)


def get_chart_layout(height=400, show_legend=True, title="", margin=None):
    """Return a consistent Plotly dark-theme layout dict."""
    base_margin = margin or dict(l=20, r=20, t=44 if title else 20, b=20)
    return dict(
        template="plotly_dark",
        height=height,
        showlegend=show_legend,
        font=dict(size=11, color="#94A3B8", family="Inter, sans-serif"),
        paper_bgcolor="#111827",
        plot_bgcolor="#111827",
        margin=base_margin,
        title=dict(text=title, font=dict(size=13, color="#F1F5F9")) if title else None,
        xaxis=dict(gridcolor="#1F2D45", showgrid=True, gridwidth=1, zeroline=False, linecolor="#1F2D45"),
        yaxis=dict(gridcolor="#1F2D45", showgrid=True, gridwidth=1, zeroline=False, linecolor="#1F2D45"),
        legend=dict(bgcolor="rgba(17,24,39,0.8)", bordercolor="#1F2D45", borderwidth=1, font=dict(size=11, color="#94A3B8")),
    )


# ============================================================================
# DATA LOADING
# ============================================================================
def load_data():
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
        if "Satisfaction Level" in df.columns:
            df["Satisfaction Level"] = df["Satisfaction Level"].fillna(1).astype(int)
        return df
    raise FileNotFoundError(f"{DATA_FILE} not found.")


def get_data_stats(df):
    return {
        "total_customers": len(df),
        "avg_spend":       df["Total Spend"].mean(),
        "avg_rating":      df["Average Rating"].mean(),
        "avg_age":         df["Age"].mean(),
        "total_revenue":   df["Total Spend"].sum(),
    }


# ============================================================================
# MODEL TRAINING & LOADING
# ============================================================================
def train_satisfaction_model(df):
    feature_cols = ["Gender","Age","Membership Type","Total Spend","Items Purchased",
                    "Average Rating","Discount Applied","Days Since Last Purchase","City Tier"]
    X = df[feature_cols].copy().dropna()
    y = df["Satisfaction Level"].astype(int)[X.index]
    model = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=5,
                                   min_samples_leaf=2, random_state=42, n_jobs=-1)
    model.fit(X, y)
    joblib.dump(model, PREDICTOR_MODEL)
    return model


def load_satisfaction_model(df):
    if os.path.exists(PREDICTOR_MODEL):
        return joblib.load(PREDICTOR_MODEL)
    return train_satisfaction_model(df)


def train_segmentation_model(df):
    feature_cols = ["Age","Membership Type","Total Spend","Items Purchased",
                    "Average Rating","Days Since Last Purchase"]
    X = df[feature_cols].copy().dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    joblib.dump(kmeans, CLUSTER_MODEL)
    joblib.dump(scaler, SCALER_MODEL)
    return kmeans, scaler


def load_segmentation_model(df):
    if os.path.exists(CLUSTER_MODEL) and os.path.exists(SCALER_MODEL):
        return joblib.load(CLUSTER_MODEL), joblib.load(SCALER_MODEL)
    return train_segmentation_model(df)


def get_cluster_labels():
    return {0: "Premium Customers", 1: "Budget Customers", 2: "Frequent Buyers", 3: "Low Engagement Users"}


# ============================================================================
# PREDICTION UTILITIES
# ============================================================================
def prepare_input_for_prediction(user_input):
    return pd.DataFrame([{
        "Gender": GENDER_MAPPING[user_input["gender"]],
        "Age": user_input["age"],
        "Membership Type": MEMBERSHIP_MAPPING[user_input["membership"]],
        "Total Spend": user_input["total_spend"],
        "Items Purchased": user_input["items_purchased"],
        "Average Rating": user_input["avg_rating"],
        "Discount Applied": 1 if user_input["discount"] else 0,
        "Days Since Last Purchase": user_input["days_since_purchase"],
        "City Tier": user_input["city_tier"],
    }])


def predict_satisfaction(model, input_df):
    prediction = model.predict(input_df)[0]
    probas     = model.predict_proba(input_df)[0]
    return {
        "prediction": SATISFACTION_MAPPING[prediction],
        "confidence": max(probas)*100,
        "class": prediction,
        "probas": probas
    }


def get_satisfaction_recommendation(result):
    if result["class"] == 2:
        return "Keep engagement high. Focus on retention and upsell opportunities."
    elif result["class"] == 1:
        return "Work on improving service quality and personalisation."
    return "Immediate action needed. Reach out with special offers and priority support."


# ============================================================================
# SEGMENTATION UTILITIES
# ============================================================================
def segment_customer(kmeans, scaler, user_input):
    input_df = pd.DataFrame([{
        "Age": user_input["age"],
        "Membership Type": user_input["membership"],
        "Total Spend": user_input["total_spend"],
        "Items Purchased": user_input["items_purchased"],
        "Average Rating": user_input["avg_rating"],
        "Days Since Last Purchase": user_input["days_since_purchase"]
    }])
    return get_cluster_labels()[kmeans.predict(scaler.transform(input_df))[0]]


def get_feature_importance(model):
    feature_cols = ["Gender","Age","Membership Type","Total Spend","Items Purchased",
                    "Average Rating","Discount Applied","Days Since Last Purchase","City Tier"]
    return pd.DataFrame({
        "Feature": feature_cols,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)


def get_business_insights(df):
    insights = {}
    top = df.groupby("Membership Type")["Total Spend"].mean().idxmax()
    insights["top_membership_avg"]    = df.groupby("Membership Type")["Total Spend"].mean().max()
    insights["top_membership"]        = {1:"Bronze",2:"Silver",3:"Gold"}[top]
    low_eng = df[df["Days Since Last Purchase"] > df["Days Since Last Purchase"].quantile(0.75)]
    insights["low_engagement_count"]  = len(low_eng)
    insights["low_engagement_pct"]    = (len(low_eng)/len(df))*100
    disc = df.groupby("Discount Applied")["Satisfaction Level"].mean()
    insights["discount_satisfaction"] = disc[1] - disc[0]
    insights["spend_satisfaction_corr"] = df["Total Spend"].corr(df["Satisfaction Level"])
    insights["satisfied_count"]   = (df["Satisfaction Level"]==2).sum()
    insights["unsatisfied_count"] = (df["Satisfaction Level"]==0).sum()
    insights["neutral_count"]     = (df["Satisfaction Level"]==1).sum()
    return insights


# ============================================================================
# SIDEBAR NAVIGATION (Stylish FA Icons + Pure HTML Render)
# ============================================================================
def render_sidebar(active_page: str):
    """Render branded sidebar with FA icons and inject global styles."""
    apply_global_styles()

    nav_items = [
        ("app",          "/",             "fa-house",        "Home"),
        ("dashboard",    "/dashboard",    "fa-chart-line",   "Dashboard"),
        ("insights",     "/insights",     "fa-lightbulb",    "Insights"),
        ("prediction",   "/prediction",   "fa-bullseye",     "Prediction"),
        ("segmentation", "/segmentation", "fa-layer-group",  "Segmentation"),
    ]

    nav_links_html = ""
    for slug, path, icon, label in nav_items:
        active_cls = "active" if slug == active_page else ""
        badge_html = "<span class='nav-badge'>ACTIVE</span>" if active_cls else ""
        nav_links_html += f"""
        <a href="{path}" target="_self" class="sb-nav-link {active_cls}">
            <i class="fa-solid {icon} nav-icon"></i>
            <span class="nav-label">{label}</span>
            {badge_html}
        </a>"""

    sidebar_full_html = f"""{FA_CDN}
    <div style="padding: 1.2rem 1rem 1rem; border-bottom: 1px solid rgba(99,102,241,0.15); margin-bottom: 0.6rem;">
        <div style="display:flex;align-items:center;gap:0.75rem;margin-bottom:0.8rem;">
            <div style="width:40px; height:40px; background: linear-gradient(135deg,#6366F1 0%,#4338CA 100%); border-radius:12px; display:flex; align-items:center; justify-content:center; box-shadow: 0 4px 14px rgba(99,102,241,0.4); flex-shrink:0;">
                <i class="fa-solid fa-chart-pie" style="color:#fff;font-size:1.1rem;"></i>
            </div>
            <div>
                <div style="color:#F1F5F9;font-size:0.95rem;font-weight:700;letter-spacing:-0.01em;line-height:1.2;">
                    Analytics Platform
                </div>
                <div style="color:#64748B;font-size:0.7rem;margin-top:0.2rem;letter-spacing:0.02em;">
                    Customer Intelligence Suite
                </div>
            </div>
        </div>
        <div style="display:inline-flex; align-items:center; gap:0.4rem; background:rgba(16,185,129,0.1); border:1px solid rgba(16,185,129,0.25); color:#34D399; font-size:0.62rem; font-weight:700; letter-spacing:0.1em; text-transform:uppercase; padding:0.22rem 0.65rem; border-radius:50px;">
            <i class="fa-solid fa-circle" style="font-size:0.4rem;"></i>
            Live &middot; Production
        </div>
    </div>

    <div style="padding: 0.6rem 1rem 0.3rem; color: #475569; font-size: 0.65rem; font-weight: 700; letter-spacing: 0.16em; text-transform: uppercase;">
        Navigation
    </div>

    {nav_links_html}

    <div style="margin:1rem 1rem; border-top:1px solid rgba(99,102,241,0.12);"></div>

    <div style="padding:0 0.6rem;">
        <div style="background: rgba(17,24,39,0.85); border: 1px solid rgba(99,102,241,0.15); border-radius: 12px; padding: 0.9rem 1rem; box-shadow: 0 4px 20px rgba(0,0,0,0.25);">
            <div style="color:#64748B; font-size:0.65rem; font-weight:700; letter-spacing:0.15em; text-transform:uppercase; margin-bottom:0.65rem; display:flex; align-items:center; gap:0.4rem;">
                <i class="fa-solid fa-circle-info" style="color:#6366F1;"></i> Platform Info
            </div>

            <div style="display:flex;justify-content:space-between;align-items:center;padding:0.3rem 0;font-size:0.75rem;border-bottom:1px solid rgba(255,255,255,0.04);">
                <span style="color:#94A3B8;display:flex;align-items:center;gap:0.45rem;">
                    <i class="fa-solid fa-robot" style="width:14px;color:#6366F1;"></i> ML Models
                </span>
                <span style="color:#F1F5F9;font-weight:600;">RF &middot; KMeans</span>
            </div>

            <div style="display:flex;justify-content:space-between;align-items:center;padding:0.3rem 0;font-size:0.75rem;border-bottom:1px solid rgba(255,255,255,0.04);">
                <span style="color:#94A3B8;display:flex;align-items:center;gap:0.45rem;">
                    <i class="fa-solid fa-database" style="width:14px;color:#6366F1;"></i> Dataset
                </span>
                <span style="color:#F1F5F9;font-weight:600;">E-Commerce</span>
            </div>

            <div style="display:flex;justify-content:space-between;align-items:center;padding:0.3rem 0;font-size:0.75rem;border-bottom:1px solid rgba(255,255,255,0.04);">
                <span style="color:#94A3B8;display:flex;align-items:center;gap:0.45rem;">
                    <i class="fa-solid fa-link" style="width:14px;color:#20BEFF;"></i> Source
                </span>
                <a href="https://www.kaggle.com/datasets/uom190346a/e-commerce-customer-behavior-dataset" target="_blank" style="color:#FFFFFF !important; -webkit-text-fill-color:#FFFFFF !important; text-decoration:none; font-weight:700; font-size:0.72rem; display:flex; align-items:center; gap:0.25rem;">
                    Kaggle <i class="fa-solid fa-arrow-up-right-from-square" style="font-size:0.6rem; color:#FFFFFF !important; -webkit-text-fill-color:#FFFFFF !important;"></i>
                </a>
            </div>

            <div style="display:flex;justify-content:space-between;align-items:center;padding:0.3rem 0;font-size:0.75rem;">
                <span style="color:#94A3B8;display:flex;align-items:center;gap:0.45rem;">
                    <i class="fa-solid fa-shield-halved" style="width:14px;color:#10B981;"></i> Status
                </span>
                <span style="color:#34D399;font-weight:700;display:flex;align-items:center;gap:0.35rem;">
                    <i class="fa-solid fa-circle-check" style="font-size:0.7rem;"></i> Ready
                </span>
            </div>
        </div>
    </div>

    <div style="padding:1.2rem 1rem 0.8rem; text-align:center;">
        <div style="color:#475569;font-size:0.68rem;letter-spacing:0.04em;">
            v1.0 &middot; scikit-learn &middot; Streamlit
        </div>
    </div>"""

    st.sidebar.html(sidebar_full_html)
