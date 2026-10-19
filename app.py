"""
Customer Behavior Analytics Platform — Landing Page
"""

import streamlit as st
from utils import (
    load_data, get_data_stats,
    load_satisfaction_model, load_segmentation_model,
    render_sidebar, FA_CDN,
)

st.set_page_config(page_title="Customer Analytics Platform", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

render_sidebar("app")

@st.cache_resource
def init_app():
    df      = load_data()
    stats   = get_data_stats(df)
    pred    = load_satisfaction_model(df)
    km, sc  = load_segmentation_model(df)
    return df, stats, pred, km, sc

df = stats = None
models_ready = False
try:
    df, stats, _, _, _ = init_app()
    models_ready = True
except Exception as e:
    st.error(f"Error loading models: {e}")

# ── Hero ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
{FA_CDN}
<div class="hero-banner">
    <div class="hero-tag">
        <i class="fa-solid fa-robot"></i> AI-Powered Customer Intelligence
    </div>
    <h1>Customer Behavior Analytics Platform</h1>
    <p class="hero-sub">
        Predict customer <strong>Satisfaction_Score</strong>, uncover customer segments,
        and generate strategic business insights across <strong>10,000 customer records</strong>
        using core behavioral and demographic attributes.
    </p>
</div>
""", unsafe_allow_html=True)

# ── Metric ribbon ─────────────────────────────────────────────────────────────
if models_ready and stats:
    c1, c2, c3, c4 = st.columns(4, gap="small")
    ribbon = [
        (c1, "fa-users",           "Total Customers",        f"{stats['total_customers']:,}",          "positive", "Active records"),
        (c2, "fa-dollar-sign",     "Average Spend",          f"${stats['avg_spend']:.2f}",             "positive", "Per customer"),
        (c3, "fa-arrow-trend-up",  "Total Revenue",          f"${stats['total_revenue']:,.0f}",        "positive", "Combined spend"),
        (c4, "fa-star",            "Avg Satisfaction Score", f"{stats['avg_score']:.2f} / 5.0",        "neutral",  "Target metric"),
    ]
    for col, icon, label, value, cls, sub in ribbon:
        with col:
            st.markdown(f"""
            <div class="pro-card">
                <span class="card-fa-icon"><i class="fa-solid {icon}"></i></span>
                <div class="card-label">{label}</div>
                <div class="card-value">{value}</div>
                <div class="card-delta {cls}">{sub}</div>
            </div>
            """, unsafe_allow_html=True)
else:
    st.info("Initialising models and loading data…")

st.divider()

# ── Platform modules ──────────────────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="section-accent"></div>
    <h2>Platform Modules</h2>
</div>
""", unsafe_allow_html=True)

fc1, fc2, fc3, fc4 = st.columns(4, gap="small")
features = [
    (fc1, "fa-chart-line",  "Business Dashboard",
     "Visualise KPIs and trends across Gender, Age, Income Level, Region, Spend, and Membership enrollment."),
    (fc2, "fa-bullseye",    "Satisfaction Predictor",
     "Forecast Satisfaction_Score (1.0–5.0) and satisfaction tier using a trained RandomForest model with confidence and recommendations."),
    (fc3, "fa-layer-group", "Customer Segmentation",
     "Discover behavioral personas through KMeans clustering based on demographics, spending habits, and membership."),
    (fc4, "fa-lightbulb",   "Business Insights",
     "Access automated recommendations surfacing membership retention drivers, income tier opportunities, and regional dynamics."),
]
for col, icon, title, desc in features:
    with col:
        st.markdown(f"""
        <div class="feature-card">
            <div class="fc-icon-wrap"><i class="fa-solid {icon}"></i></div>
            <div class="fc-title">{title}</div>
            <div class="fc-desc">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

st.divider()

# ── Dataset snapshot ──────────────────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="section-accent"></div>
    <h2>Dataset Snapshot &amp; Schema</h2>
</div>
""", unsafe_allow_html=True)

if models_ready and df is not None:
    d1, d2, d3 = st.columns(3, gap="small")

    mem_yes  = int((df["Membership"] == "Yes").sum())
    mem_no   = int((df["Membership"] == "No").sum())
    sat_high = int((df["Satisfaction_Score"] > 3.5).sum())
    sat_mod  = int(((df["Satisfaction_Score"] > 2.0) & (df["Satisfaction_Score"] <= 3.5)).sum())
    sat_low  = int((df["Satisfaction_Score"] <= 2.0).sum())

    with d1:
        st.metric("Membership Enrollment", f"{mem_yes:,} Members / {mem_no:,} Non-Members",
                  f"{mem_yes / len(df) * 100:.1f}% membership rate")
    with d2:
        st.metric("Satisfaction Breakdown", "3 Tiers",
                  f"High {sat_high:,} · Moderate {sat_mod:,} · Low {sat_low:,}")
    with d3:
        st.metric("Customer Age Profile", f"{df['Age'].mean():.1f} years avg",
                  f"Range: {df['Age'].min()}–{df['Age'].max()} years")

    # Dataset schema card
    st.markdown("""
    <div style="
        margin-top: 1.2rem;
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(17, 24, 39, 0.85) 100%);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 1rem;
    ">
        <div style="display: flex; align-items: center; gap: 1.1rem;">
            <div style="width: 46px; height: 46px; background: rgba(32, 190, 255, 0.15); border: 1px solid rgba(32, 190, 255, 0.35); border-radius: 10px; display: flex; align-items: center; justify-content: center; flex-shrink: 0;">
                <i class="fa-solid fa-database" style="color: #20BEFF; font-size: 1.2rem;"></i>
            </div>
            <div>
                <div style="color: #FFFFFF; font-size: 1rem; font-weight: 700; letter-spacing: -0.01em;">
                    Target Dataset: 10,000 Records &middot; 7 Features
                </div>
                <div style="color: #94A3B8; font-size: 0.82rem; margin-top: 0.25rem;">
                    Features: <strong style="color:#CBD5E1;">Gender, Age, Income_Level, Region, Avg_Spend, Membership</strong>
                    &middot; Target: <strong style="color:#34D399;">Satisfaction_Score</strong>
                </div>
            </div>
        </div>
        <div>
            <span style="
                display: inline-flex;
                align-items: center;
                gap: 0.55rem;
                background: linear-gradient(135deg, #6366F1 0%, #4F46E5 100%);
                border: 1px solid rgba(255,255,255,0.25);
                border-radius: 8px;
                padding: 0.65rem 1.25rem;
                box-shadow: 0 4px 14px rgba(99,102,241,0.4);
            ">
                <i class="fa-solid fa-check" style="color:#FFFFFF; font-size:0.75rem;"></i>
                <span style="color:#FFFFFF; font-weight:700; font-size:0.85rem; letter-spacing:0.01em;">
                    Active Schema
                </span>
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ── Technology stack ──────────────────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="section-accent"></div>
    <h2>Technology Stack</h2>
</div>
""", unsafe_allow_html=True)

t1, t2, t3, t4 = st.columns(4, gap="small")
stack = [
    (t1, "fa-desktop",    "Frontend",    ["Streamlit 1.53+", "Plotly Visualizations", "Custom Dark Theme"]),
    (t2, "fa-gear",       "Backend",     ["Python 3.9+", "Pandas & NumPy", "Joblib Model Persistence"]),
    (t3, "fa-robot",      "ML / AI",     ["scikit-learn", "RandomForest (Score & Class)", "KMeans Clustering"]),
    (t4, "fa-cloud",      "Deployment",  ["Streamlit Cloud", "Production Ready", "10,000 Records"]),
]
for col, icon, title, items in stack:
    rows = "".join(f'<div style="color:#4B5A72;font-size:0.78rem;padding:0.18rem 0;display:flex;align-items:center;gap:0.4rem;"><i class="fa-solid fa-check" style="font-size:0.6rem;color:#6366F1;"></i>{i}</div>' for i in items)
    with col:
        st.markdown(f"""
        <div class="feature-card" style="padding:1.1rem 1.2rem;">
            <div class="fc-icon-wrap"><i class="fa-solid {icon}"></i></div>
            <div class="fc-title">{title}</div>
            {rows}
        </div>
        """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="
    margin-top:2rem;padding:1.5rem 2rem;
    background:linear-gradient(135deg,#111827 0%,#1A2234 100%);
    border:1px solid #1F2D45;border-radius:14px;text-align:center;
">
    <div style="color:#F1F5F9;font-size:1rem;font-weight:600;margin-bottom:0.4rem;">
        <i class="fa-solid fa-chart-pie" style="color:#6366F1;margin-right:0.4rem;"></i>
        Customer Behavior Analytics Platform
    </div>
    <div style="color:#64748B;font-size:0.8rem;line-height:1.6;">
        ML-powered analytics &middot; Predicting Satisfaction_Score &middot; 10,000 Records
    </div>
    <div style="margin-top:0.8rem;display:flex;gap:0.5rem;justify-content:center;flex-wrap:wrap;">
        <span class="badge badge-indigo"><i class="fa-solid fa-robot"></i> AI Powered</span>
        <span class="badge badge-green"><i class="fa-solid fa-circle-check"></i> Production Ready</span>
        <span class="badge badge-amber"><i class="fa-solid fa-chart-bar"></i> Real-time Analytics</span>
    </div>
</div>
""", unsafe_allow_html=True)