"""
Utility functions for model training, loading, data processing, and enterprise styling.
Dataset schema: Gender, Age, Income_Level, Region, Avg_Spend, Membership, Satisfaction_Score.
"""
import streamlit as st
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
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

GENDER_MAPPING     = {"F": 0, "M": 1, "Other": 2}
GENDER_REVERSE     = {0: "F", 1: "M", 2: "Other"}
INCOME_MAPPING     = {"Low": 0, "Medium": 1, "High": 2}
INCOME_REVERSE     = {0: "Low", 1: "Medium", 2: "High"}
REGION_MAPPING     = {"Rural": 0, "Semi-Urban": 1, "Urban": 2}
REGION_REVERSE     = {0: "Rural", 1: "Semi-Urban", 2: "Urban"}
MEMBERSHIP_MAPPING = {"No": 0, "Yes": 1}
MEMBERSHIP_REVERSE = {0: "No", 1: "Yes"}

SATISFACTION_MAPPING = {0: "Low Satisfaction", 1: "Moderate Satisfaction", 2: "High Satisfaction"}

PRED_FEATURE_COLS = ["Age", "Gender_Enc", "Income_Enc", "Region_Enc", "Avg_Spend", "Membership_Enc"]
SEG_FEATURE_COLS  = ["Age", "Avg_Spend", "Income_Level"]

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

/* ── Forceful Button Text & Style ─────────────────────────────────────────── */
button, button *, button p, button span, [data-testid*="Button"], [data-testid*="Button"] * {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.02em !important;
}
div[data-testid="stButton"] > button,
div[data-testid="stButton"] button {
    background: linear-gradient(135deg, #6366F1 0%, #4F46E5 100%) !important;
    border: 1px solid rgba(255, 255, 255, 0.25) !important;
    border-radius: 10px !important;
    padding: 0.65rem 1.4rem !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 14px rgba(99,102,241,0.4) !important;
    min-height: 44px !important;
}
div[data-testid="stButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(99,102,241,0.6) !important;
    filter: brightness(1.12) !important;
}

/* ── Reusable Component Classes ────────────────────────────────────────────── */
.pro-card {
    background: linear-gradient(135deg, #111827 0%, #1A2234 100%);
    border: 1px solid var(--border-soft);
    border-radius: 14px;
    padding: 1.25rem 1.4rem;
    text-align: center;
    transition: transform 0.22s ease, box-shadow 0.22s ease, border-color 0.22s ease;
}
.pro-card:hover { transform: translateY(-3px); box-shadow: 0 12px 36px rgba(0,0,0,0.45); border-color: rgba(99,102,241,0.35); }
.pro-card .card-fa-icon { font-size: 1.2rem; color: var(--accent); margin-bottom: 0.6rem; display: block; }
.pro-card .card-label { color: var(--text-muted); font-size: 0.68rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 0.5rem; }
.pro-card .card-value { color: var(--text-primary); font-size: 1.85rem; font-weight: 700; letter-spacing: -0.03em; line-height: 1.1; margin-bottom: 0.4rem; }
.pro-card .card-delta { font-size: 0.75rem; font-weight: 500; }
.card-delta.positive { color: #34D399; }
.card-delta.neutral  { color: #FCD34D; }
.card-delta.info     { color: #60A5FA; }

.hero-banner {
    background: linear-gradient(135deg, #0F172A 0%, #1E1B4B 55%, #0F172A 100%);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 18px;
    padding: 2.2rem 2.6rem;
    margin-bottom: 1.75rem;
    position: relative; overflow: hidden;
}
.hero-tag {
    display: inline-flex; align-items: center; gap: 0.45rem;
    background: rgba(99,102,241,0.14);
    border: 1px solid rgba(99,102,241,0.32);
    color: #A5B4FC;
    font-size: 0.7rem; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase;
    padding: 0.22rem 0.7rem; border-radius: 50px; margin-bottom: 0.85rem;
}
.hero-sub { color: var(--text-muted); font-size: 0.92rem; line-height: 1.65; max-width: 580px; }

.section-header { display: flex; align-items: center; gap: 0.6rem; margin: 1.6rem 0 1rem 0; }
.section-accent { width: 4px; height: 20px; background: linear-gradient(180deg, #6366F1 0%, #4F46E5 100%); border-radius: 2px; flex-shrink: 0; }

.feature-card {
    background: var(--bg-card); border: 1px solid var(--border); border-radius: 14px;
    padding: 1.4rem 1.5rem; height: 100%; transition: all 0.22s ease;
}
.feature-card:hover { border-color: rgba(99,102,241,0.4); transform: translateY(-2px); box-shadow: 0 8px 30px rgba(0,0,0,0.35); }
.feature-card .fc-icon-wrap {
    width: 40px; height: 40px; background: rgba(99,102,241,0.1);
    border: 1px solid rgba(99,102,241,0.2); border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    margin-bottom: 0.85rem; font-size: 0.95rem; color: var(--accent);
}
.feature-card .fc-title { color: var(--text-primary); font-size: 0.95rem; font-weight: 600; margin-bottom: 0.5rem; }
.feature-card .fc-desc  { color: var(--text-muted); font-size: 0.82rem; line-height: 1.6; }

.insight-panel {
    background: var(--bg-card); border: 1px solid var(--border-soft); border-radius: 12px;
    padding: 1.3rem 1.5rem; margin-bottom: 1rem;
}
.insight-panel .ip-num { color: var(--accent); font-size: 0.68rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 0.35rem; }
.insight-panel h4 { color: var(--text-primary) !important; font-size: 0.97rem !important; font-weight: 600 !important; margin: 0 0 0.55rem 0 !important; }

.rec-card {
    background: var(--bg-card); border: 1px solid var(--border-soft);
    border-top: 3px solid var(--accent); border-radius: 12px;
    padding: 1.2rem 1.4rem; margin-bottom: 0.85rem;
}
.rec-card .rc-title { color: var(--text-primary); font-size: 0.95rem; font-weight: 600; margin-bottom: 0.45rem; }
.rec-card .rc-desc  { color: var(--text-soft); font-size: 0.82rem; line-height: 1.55; margin-bottom: 0.6rem; }
.rc-meta { display: flex; gap: 0.4rem; flex-wrap: wrap; align-items: center; }

/* Redesigned Rich Segment Card */
.seg-card-rich {
    background: var(--bg-card);
    border: 1px solid var(--border-soft);
    border-radius: 16px;
    padding: 1.4rem 1.5rem;
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    transition: transform 0.22s ease, box-shadow 0.22s ease, border-color 0.22s ease;
}
.seg-card-rich:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 32px rgba(0,0,0,0.4);
    border-color: rgba(99,102,241,0.4);
}
.seg-card-rich .sc-header {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 1rem;
    padding-bottom: 0.85rem;
    border-bottom: 1px solid var(--border);
}
.seg-card-rich .sc-stripe { width: 4px; height: 32px; border-radius: 2px; flex-shrink: 0; }
.seg-card-rich .sc-icon-wrap {
    width: 36px; height: 36px; border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.95rem; flex-shrink: 0;
}
.seg-card-rich .sc-title { color: var(--text-primary); font-size: 1.05rem; font-weight: 700; line-height: 1.2; }
.seg-card-rich .sc-badge { font-size: 0.62rem; font-weight: 700; padding: 0.15rem 0.5rem; border-radius: 50px; text-transform: uppercase; letter-spacing: 0.06em; margin-top: 0.25rem; display: inline-block; }
.seg-card-rich .sc-stats-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.65rem 0.9rem;
    margin-bottom: 1rem;
}
.seg-card-rich .sc-stat-label { color: var(--text-muted); font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.15rem; }
.seg-card-rich .sc-stat-val { color: var(--text-primary); font-size: 0.98rem; font-weight: 700; }
.seg-card-rich .sc-desc-box {
    background: rgba(10, 14, 23, 0.6);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.75rem 0.9rem;
    color: var(--text-soft);
    font-size: 0.8rem;
    line-height: 1.5;
    margin-bottom: 0.85rem;
}
.seg-card-rich .sc-rec-box {
    background: rgba(99,102,241,0.06);
    border-left: 3px solid var(--accent);
    border-radius: 0 8px 8px 0;
    padding: 0.65rem 0.85rem;
    font-size: 0.78rem;
    color: #E2E8F0;
    line-height: 1.45;
}

.badge { display: inline-block; padding: 0.18rem 0.55rem; border-radius: 50px; font-size: 0.68rem; font-weight: 600; letter-spacing: 0.04em; }
.badge-green  { background: rgba(16,185,129,0.14);  color: #34D399; border: 1px solid rgba(16,185,129,0.28); }
.badge-amber  { background: rgba(245,158,11,0.14);  color: #FCD34D; border: 1px solid rgba(245,158,11,0.28); }
.badge-red    { background: rgba(239,68,68,0.14);   color: #FCA5A5; border: 1px solid rgba(239,68,68,0.28); }
.badge-indigo { background: rgba(99,102,241,0.14);  color: #A5B4FC; border: 1px solid rgba(99,102,241,0.28); }

.result-card { border-radius: 14px; padding: 1.4rem; text-align: center; }
.result-card .rc-label  { font-size: 0.7rem; font-weight: 600; letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 0.5rem; opacity: 0.8; }
.result-card .rc-value  { font-size: 1.55rem; font-weight: 700; letter-spacing: -0.02em; }
</style>
"""

def apply_global_styles():
    st.markdown(FA_CDN + GLOBAL_CSS, unsafe_allow_html=True)

def get_chart_layout(height=400, show_legend=True, title="", margin=None):
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
        return pd.read_csv(DATA_FILE)
    raise FileNotFoundError(f"{DATA_FILE} not found.")

def get_data_stats(df):
    return {
        "total_customers": len(df),
        "avg_spend":       df["Avg_Spend"].mean(),
        "avg_score":       df["Satisfaction_Score"].mean(),
        "avg_age":         df["Age"].mean(),
        "total_revenue":   df["Avg_Spend"].sum(),
        "membership_pct":  (df["Membership"] == "Yes").mean() * 100,
    }

# ============================================================================
# SATISFACTION PREDICTOR MODEL TRAINING & LOADING
# ============================================================================
def train_satisfaction_model(df):
    X = df[PRED_FEATURE_COLS].copy()
    y_class = df["Satisfaction_Class"].astype(int)
    y_score = df["Satisfaction_Score"].astype(float)

    clf = RandomForestClassifier(n_estimators=150, max_depth=12, min_samples_split=4,
                                 min_samples_leaf=2, random_state=42, n_jobs=-1)
    clf.fit(X, y_class)

    reg = RandomForestRegressor(n_estimators=150, max_depth=12, min_samples_split=4,
                                min_samples_leaf=2, random_state=42, n_jobs=-1)
    reg.fit(X, y_score)

    bundle = {"classifier": clf, "regressor": reg}
    try: joblib.dump(bundle, PREDICTOR_MODEL)
    except Exception: pass
    return bundle

def load_satisfaction_model(df):
    if os.path.exists(PREDICTOR_MODEL):
        try:
            bundle = joblib.load(PREDICTOR_MODEL)
            if isinstance(bundle, dict) and "classifier" in bundle:
                return bundle
        except Exception: pass
    return train_satisfaction_model(df)

# ============================================================================
# K-MEANS SEGMENTATION PIPELINE (Age, Avg_Spend, Income_Level)
# ============================================================================
def build_segmentation_pipeline(n_clusters=4):
    """
    Builds a robust, leakage-free Scikit-learn Pipeline:
    - Age & Avg_Spend: Standardized via StandardScaler
    - Income_Level: Ordinally encoded (Low=0, Medium=1, High=2) and Standardized
    - Clustered via KMeans(k=4, random_state=42)
    Excludes Gender & Membership to avoid artificial identity clustering.
    Excludes Satisfaction_Score to prevent target leakage and allow independent comparison.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['Age', 'Avg_Spend']),
            ('ord', Pipeline([
                ('ordinal', OrdinalEncoder(categories=[['Low', 'Medium', 'High']])),
                ('scaler', StandardScaler())
            ]), ['Income_Level'])
        ]
    )
    return Pipeline([
        ('preprocessor', preprocessor),
        ('cluster', KMeans(n_clusters=n_clusters, random_state=42, n_init=10))
    ])

def train_segmentation_model(df):
    pipeline = build_segmentation_pipeline(n_clusters=4)
    pipeline.fit(df[SEG_FEATURE_COLS])
    try:
        joblib.dump(pipeline, CLUSTER_MODEL)
        joblib.dump(pipeline.named_steps['preprocessor'], SCALER_MODEL)
    except Exception: pass
    return pipeline, pipeline.named_steps['preprocessor']

def load_segmentation_model(df):
    if os.path.exists(CLUSTER_MODEL) and os.path.exists(SCALER_MODEL):
        try:
            pipeline = joblib.load(CLUSTER_MODEL)
            if hasattr(pipeline, "named_steps") and "preprocessor" in getattr(pipeline, "named_steps", {}):
                return pipeline, pipeline.named_steps['preprocessor']
        except Exception: pass
    return train_segmentation_model(df)

# ============================================================================
# DYNAMIC CLUSTER PERSONA PROFILING
# ============================================================================
def get_dynamic_cluster_profiles(df, cluster_col="Cluster"):
    """
    Dynamically discovers and profiles clusters based on actual data statistics:
    - Evaluates spend, age, income distributions, satisfaction, and membership rates.
    - Dynamically generates interpretable persona names, badges, colors, and business recommendations.
    - Does NOT hardcode static cluster IDs so results are consistent and data-driven.
    """
    overall_spend = df["Avg_Spend"].mean()
    overall_age   = df["Age"].mean()

    cluster_stats = []
    for c in sorted(df[cluster_col].unique()):
        sub = df[df[cluster_col] == c]
        cnt = len(sub)
        pct = (cnt / len(df)) * 100
        avg_spd = sub["Avg_Spend"].mean()
        avg_age = sub["Age"].mean()
        avg_sat = sub["Satisfaction_Score"].mean()
        mem_pct = (sub["Membership"] == "Yes").mean() * 100
        inc_dist = sub["Income_Level"].value_counts(normalize=True).to_dict()
        reg_dist = sub["Region"].value_counts(normalize=True).to_dict()

        cluster_stats.append({
            "cluster_id": c,
            "count": cnt,
            "pct": pct,
            "avg_spend": avg_spd,
            "avg_age": avg_age,
            "avg_sat": avg_sat,
            "mem_pct": mem_pct,
            "inc_dist": inc_dist,
            "reg_dist": reg_dist,
        })

    assigned = {}
    remaining = cluster_stats.copy()

    # 1. Highest spend cluster -> High-Value Spenders
    highest_spend = max(remaining, key=lambda x: x["avg_spend"])
    h_spd = round(highest_spend["avg_spend"], 1)
    assigned[highest_spend["cluster_id"]] = {
        "name": "High-Value Spenders",
        "badge": "Revenue Driver",
        "color": "#10B981",
        "icon": "fa-crown",
        "desc": f"Top spenders averaging ${h_spd:.1f} per transaction (significantly above the overall average of ${overall_spend:.1f}).",
        "rec": "Prioritize VIP concierge perks, premium loyalty tiers, and early access to high-ticket offerings."
    }
    remaining = [x for x in remaining if x["cluster_id"] != highest_spend["cluster_id"]]

    # 2. Highest proportion of High Income -> Affluent High-Income
    high_inc = max(remaining, key=lambda x: x["inc_dist"].get("High", 0))
    h_inc_pct = round(high_inc["inc_dist"].get("High", 0) * 100)
    assigned[high_inc["cluster_id"]] = {
        "name": "Affluent High-Income",
        "badge": "High Purchasing Power",
        "color": "#6366F1",
        "icon": "fa-wallet",
        "desc": f"{h_inc_pct}% high-income customers with strong purchasing capacity and dependable engagement.",
        "rec": "Target with premium subscription tiers, luxury partnerships, and personalized service guarantees."
    }
    remaining = [x for x in remaining if x["cluster_id"] != high_inc["cluster_id"]]

    # 3. Among remaining, split by age: Younger vs Mature
    younger = min(remaining, key=lambda x: x["avg_age"])
    older   = max(remaining, key=lambda x: x["avg_age"])

    y_age = round(younger["avg_age"], 1)
    y_spd = round(younger["avg_spend"], 1)
    assigned[younger["cluster_id"]] = {
        "name": "Young Emerging Shoppers",
        "badge": "Growth Cohort",
        "color": "#06B6D4",
        "icon": "fa-bolt",
        "desc": f"Younger cohort (avg age {y_age:.1f}) with budget-to-moderate spending (${y_spd:.1f}) and strong digital potential.",
        "rec": "Engage via mobile-first flash promotions, flexible payment plans, and gamified loyalty rewards."
    }

    o_age = round(older["avg_age"], 1)
    o_spd = round(older["avg_spend"], 1)
    assigned[older["cluster_id"]] = {
        "name": "Mature Steady Buyers",
        "badge": "Stable Mainstay",
        "color": "#F59E0B",
        "icon": "fa-user-check",
        "desc": f"Mature cohort (avg age {o_age:.1f}) with consistent spending (${o_spd:.1f}) and dependable membership retention.",
        "rec": "Retain with straightforward value messaging, dependable telephone support, and reliable reorder incentives."
    }

    return assigned, cluster_stats

def get_cluster_labels(df=None):
    """Fallback or dynamic lookup for cluster labels."""
    if df is not None and "Cluster" in df.columns:
        assigned, _ = get_dynamic_cluster_profiles(df)
        return {cid: info["name"] for cid, info in assigned.items()}
    return {
        0: "Affluent High-Income",
        1: "Mature Steady Buyers",
        2: "Young Emerging Shoppers",
        3: "High-Value Spenders"
    }

def segment_customer(pipeline, user_input):
    """
    Transforms a single customer input using the exact fitted pipeline and predicts cluster.
    Accepts: age (int), income (str: Low/Medium/High), avg_spend (float).
    """
    input_df = pd.DataFrame([{
        "Age":          user_input["age"],
        "Avg_Spend":    user_input["avg_spend"],
        "Income_Level": user_input["income"],
    }])
    cluster_id = int(pipeline.predict(input_df)[0])
    return cluster_id

# ============================================================================
# PREDICTION UTILITIES
# ============================================================================
def prepare_input_for_prediction(user_input):
    return pd.DataFrame([{
        "Age":            user_input["age"],
        "Gender_Enc":     GENDER_MAPPING[user_input["gender"]],
        "Income_Enc":     INCOME_MAPPING[user_input["income"]],
        "Region_Enc":     REGION_MAPPING[user_input["region"]],
        "Avg_Spend":      user_input["avg_spend"],
        "Membership_Enc": MEMBERSHIP_MAPPING[user_input["membership"]],
    }])

def predict_satisfaction(model_bundle, input_df):
    clf = model_bundle["classifier"]
    reg = model_bundle["regressor"]
    pred_class = clf.predict(input_df)[0]
    probas     = clf.predict_proba(input_df)[0]
    score      = float(reg.predict(input_df)[0])
    score_clamped = max(1.0, min(5.0, score))

    return {
        "predicted_score": score_clamped,
        "prediction":      SATISFACTION_MAPPING[pred_class],
        "confidence":      max(probas) * 100,
        "class":           pred_class,
        "probas":          probas,
    }

def get_satisfaction_recommendation(result):
    if result["class"] == 2:
        return "Customer is highly satisfied. Recommended action: reward with VIP perks, referral bonuses, and early access to high-value offers."
    elif result["class"] == 1:
        return "Moderate satisfaction. Recommended action: engage with personalized membership upgrades, targeted discounts, and service surveys."
    return "Low satisfaction alert! Recommended action: proactive retention outreach, dedicated customer support resolution, and compensatory loyalty incentives."

def get_feature_importance(model_bundle):
    clf = model_bundle["classifier"]
    return pd.DataFrame({
        "Feature": PRED_FEATURE_COLS,
        "Importance": clf.feature_importances_
    }).sort_values("Importance", ascending=False)

def get_business_insights(df):
    insights = {}
    total = len(df)
    insights["total_customers"]   = total
    insights["membership_count"]  = int((df["Membership"] == "Yes").sum())
    insights["membership_pct"]    = (df["Membership"] == "Yes").mean() * 100
    insights["high_sat_count"]    = int((df["Satisfaction_Class"] == 2).sum())
    insights["mod_sat_count"]     = int((df["Satisfaction_Class"] == 1).sum())
    insights["low_sat_count"]     = int((df["Satisfaction_Class"] == 0).sum())
    insights["low_sat_pct"]       = (insights["low_sat_count"] / total) * 100
    insights["spend_sat_corr"]    = float(df["Avg_Spend"].corr(df["Satisfaction_Score"]))

    top_region = df.groupby("Region")["Avg_Spend"].mean().idxmax()
    insights["top_region"]        = top_region
    insights["top_region_spend"]  = float(df.groupby("Region")["Avg_Spend"].mean().max())

    top_income = df.groupby("Income_Level")["Avg_Spend"].mean().idxmax()
    insights["top_income"]        = top_income
    insights["top_income_spend"]  = float(df.groupby("Income_Level")["Avg_Spend"].mean().max())

    mem_spend_diff = float(df[df["Membership"] == "Yes"]["Avg_Spend"].mean() - df[df["Membership"] == "No"]["Avg_Spend"].mean())
    insights["mem_spend_diff"]    = mem_spend_diff

    mem_sat_diff = float(df[df["Membership"] == "Yes"]["Satisfaction_Score"].mean() - df[df["Membership"] == "No"]["Satisfaction_Score"].mean())
    insights["mem_sat_diff"]      = mem_sat_diff

    return insights

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================
def render_sidebar(active_page: str):
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
                <i class="fa-solid fa-circle-info" style="color:#6366F1;"></i> Platform Overview
            </div>

            <div style="display:flex;justify-content:space-between;align-items:center;padding:0.3rem 0;font-size:0.75rem;border-bottom:1px solid rgba(255,255,255,0.04);">
                <span style="color:#94A3B8;display:flex;align-items:center;gap:0.45rem;">
                    <i class="fa-solid fa-database" style="width:14px;color:#6366F1;"></i> Customers
                </span>
                <span style="color:#F1F5F9;font-weight:600;">10,000</span>
            </div>

            <div style="display:flex;justify-content:space-between;align-items:center;padding:0.3rem 0;font-size:0.75rem;border-bottom:1px solid rgba(255,255,255,0.04);">
                <span style="color:#94A3B8;display:flex;align-items:center;gap:0.45rem;">
                    <i class="fa-solid fa-layer-group" style="width:14px;color:#6366F1;"></i> Segments
                </span>
                <span style="color:#F1F5F9;font-weight:600;">4 Personas</span>
            </div>

            <div style="display:flex;justify-content:space-between;align-items:center;padding:0.3rem 0;font-size:0.75rem;border-bottom:1px solid rgba(255,255,255,0.04);">
                <span style="color:#94A3B8;display:flex;align-items:center;gap:0.45rem;">
                    <i class="fa-solid fa-robot" style="width:14px;color:#6366F1;"></i> ML Pipeline
                </span>
                <span style="color:#F1F5F9;font-weight:600;">KMeans &middot; RF</span>
            </div>

            <div style="display:flex;justify-content:space-between;align-items:center;padding:0.3rem 0;font-size:0.75rem;">
                <span style="color:#94A3B8;display:flex;align-items:center;gap:0.45rem;">
                    <i class="fa-solid fa-shield-halved" style="width:14px;color:#10B981;"></i> Status
                </span>
                <span style="color:#34D399;font-weight:700;display:flex;align-items:center;gap:0.35rem;">
                    <i class="fa-solid fa-circle-check" style="font-size:0.7rem;"></i> Active
                </span>
            </div>
        </div>
    </div>

    <div style="padding:1.2rem 1rem 0.8rem; text-align:center;">
        <div style="color:#475569;font-size:0.68rem;letter-spacing:0.04em;">
            v2.0 &middot; Pipeline &middot; Streamlit
        </div>
    </div>"""

    st.sidebar.html(sidebar_full_html)
