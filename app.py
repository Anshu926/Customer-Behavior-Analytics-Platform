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
        Advanced customer intelligence powered by machine learning.
        Predict satisfaction, discover segments, and unlock actionable
        insights — all in one enterprise-grade platform.
    </p>
</div>
""", unsafe_allow_html=True)

# ── Metric ribbon ─────────────────────────────────────────────────────────────
if models_ready and stats:
    c1, c2, c3, c4 = st.columns(4, gap="small")
    ribbon = [
        (c1, "fa-users",           "Total Customers", f"{stats['total_customers']:,}",    "positive", "Active records"),
        (c2, "fa-dollar-sign",     "Average Spend",   f"${stats['avg_spend']:.2f}",       "positive", "Per customer"),
        (c3, "fa-arrow-trend-up",  "Total Revenue",   f"${stats['total_revenue']:,.0f}",  "positive", "Lifetime value"),
        (c4, "fa-star",            "Average Rating",  f"{stats['avg_rating']:.2f} / 5.0", "neutral",  "Customer score"),
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
     "Visualize KPIs and customer metrics across multiple dimensions. Monitor satisfaction trends in real time."),
    (fc2, "fa-bullseye",    "Satisfaction Predictor",
     "Forecast customer satisfaction using a trained RandomForest model with confidence scores and recommendations."),
    (fc3, "fa-layer-group", "Customer Segmentation",
     "Discover behavioral personas through KMeans clustering. Understand spending patterns for each segment."),
    (fc4, "fa-lightbulb",   "Business Insights",
     "Access data-driven recommendations and analytics. Surface growth opportunities and retention risks automatically."),
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
    <h2>Dataset Snapshot & Reference</h2>
</div>
""", unsafe_allow_html=True)

if models_ready and df is not None:
    d1, d2, d3 = st.columns(3, gap="small")
    membership_dist  = df["Membership Type"].value_counts().sort_index()
    satisfaction_dist = df["Satisfaction Level"].value_counts()
    sat = satisfaction_dist.get

    with d1:
        st.metric("Membership Tiers", "3 Tiers",
                  f"Bronze {membership_dist.get(1,0)} · Silver {membership_dist.get(2,0)} · Gold {membership_dist.get(3,0)}")
    with d2:
        st.metric("Satisfaction Breakdown", "3 Levels",
                  f"Satisfied {sat(2,0)} · Neutral {sat(1,0)} · Unsatisfied {sat(0,0)}")
    with d3:
        st.metric("Average Customer Age", f"{df['Age'].mean():.1f} years",
                  f"Range: {df['Age'].min()}–{df['Age'].max()}")

    # Dataset attribution card
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
                <svg width="22" height="22" viewBox="0 0 24 24" fill="#20BEFF" xmlns="http://www.w3.org/2000/svg">
                    <path d="M18.825 23.859c-.022.046-.054.084-.092.115s-.084.053-.133.065c-.05.011-.101.011-.151 0s-.098-.033-.139-.065l-6.845-5.69-2.733 2.623v2.858a.24.24 0 0 1-.07.17.24.24 0 0 1-.17.07H5.25a.24.24 0 0 1-.17-.07.24.24 0 0 1-.07-.17V.24A.24.24 0 0 1 5.08.07.24.24 0 0 1 5.25 0h3.243c.064 0 .125.025.17.07.045.045.07.106.07.17v14.168l8.98-9.053a.35.35 0 0 1 .253-.105h4.154a.24.24 0 0 1 .184.086.24.24 0 0 1 .054.198l-7.79 7.747 8.358 10.308a.24.24 0 0 1 .054.207.24.24 0 0 1-.153.16z"/>
                </svg>
            </div>
            <div>
                <div style="color: #FFFFFF; font-size: 1rem; font-weight: 700; letter-spacing: -0.01em;">
                    E-Commerce Customer Behavior Dataset
                </div>
                <div style="color: #94A3B8; font-size: 0.82rem; margin-top: 0.25rem;">
                    Source: <strong style="color: #FFFFFF; font-weight: 600;">Kaggle</strong> (<span style="color:#CBD5E1;">@uom190346a</span>) &middot; Includes demographics, spend history, membership tiers, and satisfaction metrics.
                </div>
            </div>
        </div>
        <div>
            <a href="https://www.kaggle.com/datasets/uom190346a/e-commerce-customer-behavior-dataset" target="_blank" class="kaggle-btn" style="
                display: inline-flex !important;
                align-items: center !important;
                gap: 0.55rem !important;
                background: linear-gradient(135deg, #6366F1 0%, #4F46E5 100%) !important;
                border: 1px solid rgba(255, 255, 255, 0.25) !important;
                border-radius: 8px !important;
                padding: 0.65rem 1.25rem !important;
                text-decoration: none !important;
                box-shadow: 0 4px 14px rgba(99, 102, 241, 0.4) !important;
            ">
                <i class="fa-solid fa-arrow-up-right-from-square" style="color: #FFFFFF !important; font-size: 0.75rem; -webkit-text-fill-color: #FFFFFF !important;"></i>
                <span style="color: #FFFFFF !important; font-weight: 700 !important; font-size: 0.85rem !important; -webkit-text-fill-color: #FFFFFF !important; letter-spacing: 0.01em;">View Dataset on Kaggle</span>
            </a>
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
    (t1, "fa-desktop",    "Frontend",    ["Streamlit 1.53+", "Plotly Charts", "Custom Design System"]),
    (t2, "fa-gear",       "Backend",     ["Python 3.9+", "Pandas & NumPy", "Joblib Persistence"]),
    (t3, "fa-robot",      "ML / AI",     ["scikit-learn", "RandomForest Classifier", "KMeans Clustering"]),
    (t4, "fa-cloud",      "Deployment",  ["Streamlit Cloud Ready", "Production Grade", "Auto Model Training"]),
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
        ML-powered analytics &middot; RandomForest &middot; KMeans &middot; Production Ready
    </div>
    <div style="margin-top:0.8rem;display:flex;gap:0.5rem;justify-content:center;flex-wrap:wrap;">
        <span class="badge badge-indigo"><i class="fa-solid fa-robot"></i> AI Powered</span>
        <span class="badge badge-green"><i class="fa-solid fa-circle-check"></i> Production Ready</span>
        <span class="badge badge-amber"><i class="fa-solid fa-chart-bar"></i> Real-time Analytics</span>
    </div>
</div>
""", unsafe_allow_html=True)