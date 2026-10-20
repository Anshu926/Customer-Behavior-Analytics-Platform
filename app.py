"""
Customer Behavior Analytics Platform — Landing Page
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

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
        Advanced consumer analytics powered by machine learning —
        built on <strong>10,000 comprehensive customer records</strong> across 14 demographic,
        transactional, and behavioral dimensions. Predict satisfaction, discover actionable
        segments, and optimize revenue in one enterprise platform.
    </p>
</div>
""", unsafe_allow_html=True)

# ── Metric ribbon ─────────────────────────────────────────────────────────────
if models_ready and stats:
    c1, c2, c3, c4 = st.columns(4, gap="small")
    ribbon = [
        (c1, "fa-users",           "Total Customers",        f"{stats['total_customers']:,}",          "positive", "Active records"),
        (c2, "fa-dollar-sign",     "Average Spend",          f"${stats['avg_spend']:.2f}",             "positive", "Per customer"),
        (c3, "fa-arrow-trend-up",  "Total Revenue",          f"${stats['total_revenue']:,.0f}",        "positive", "Lifetime value"),
        (c4, "fa-star",            "Avg Satisfaction Score", f"{stats['avg_score']:.2f} / 5.0",        "neutral",  "Customer score"),
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
     "Visualise KPIs, basket size, discount usage, recency patterns, and revenue distributions across product categories and geographic regions."),
    (fc2, "fa-bullseye",    "Satisfaction Predictor",
     "Forecast customer satisfaction using a trained RandomForest model with 11 behavioral and demographic inputs, complete with probability scores and strategic action plans."),
    (fc3, "fa-layer-group", "Customer Segmentation",
     "Discover behavioral personas — High-Value Champions, Engaged Members, Prospect Shoppers, and At-Risk Inactive — with instant new customer classification."),
    (fc4, "fa-lightbulb",   "Business Insights",
     "Access automated diagnostic intelligence. Uncover category revenue drivers, loyalty impact, discount sensitivity, and churn risk indicators."),
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

# ── Dataset Information ───────────────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="section-accent"></div>
    <h2>Dataset Information &amp; Files</h2>
</div>
""", unsafe_allow_html=True)

if models_ready and df is not None:
    # 1. Two Data Sources (Kaggle & Zenodo)
    s_col1, s_col2 = st.columns(2, gap="medium")

    with s_col1:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(17, 24, 39, 0.9) 100%);
            border: 1px solid rgba(32, 190, 255, 0.28);
            border-radius: 12px;
            padding: 1.25rem 1.4rem;
            margin-bottom: 1rem;
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        ">
            <div>
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.75rem;">
                    <span style="background:rgba(32, 190, 255, 0.15); border:1px solid rgba(32, 190, 255, 0.35); color:#38BDF8; font-size:0.68rem; font-weight:700; letter-spacing:0.08em; text-transform:uppercase; padding:0.25rem 0.65rem; border-radius:6px;">
                        <i class="fa-brands fa-kaggle" style="margin-right:0.35rem;"></i> Data Source 1 &middot; Kaggle
                    </span>
                    <span style="color:#94A3B8; font-size:0.75rem; font-weight:600;">@uom190346a</span>
                </div>
                <div style="color:#F1F5F9; font-size:1.02rem; font-weight:700; margin-bottom:0.35rem;">
                    E-Commerce Customer Behavior Dataset
                </div>
                <div style="color:#94A3B8; font-size:0.83rem; line-height:1.5; margin-bottom:1rem;">
                    Benchmark consumer shopping dataset capturing purchasing dynamics, payment channel preferences, basket volumes, loyalty enrollment, and satisfaction scores across demographics.
                </div>
            </div>
            <div>
                <a href="https://www.kaggle.com/datasets/uom190346a/e-commerce-customer-behavior-dataset" target="_blank" style="
                    display: inline-flex;
                    align-items: center;
                    justify-content: center;
                    gap: 0.5rem;
                    width: 100%;
                    background: rgba(32, 190, 255, 0.12);
                    border: 1px solid rgba(32, 190, 255, 0.35);
                    color: #FFFFFF !important;
                    -webkit-text-fill-color: #FFFFFF !important;
                    border-radius: 8px;
                    padding: 0.6rem 1rem;
                    font-size: 0.82rem;
                    font-weight: 600;
                    text-decoration: none;
                    transition: all 0.2s ease;
                ">
                    <i class="fa-brands fa-kaggle" style="color: #20BEFF;"></i>
                    <span>Open Kaggle Source</span>
                    <i class="fa-solid fa-arrow-up-right-from-square" style="font-size: 0.65rem; color: #94A3B8;"></i>
                </a>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with s_col2:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(17, 24, 39, 0.9) 100%);
            border: 1px solid rgba(99, 102, 241, 0.28);
            border-radius: 12px;
            padding: 1.25rem 1.4rem;
            margin-bottom: 1rem;
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        ">
            <div>
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.75rem;">
                    <span style="background:rgba(99, 102, 241, 0.15); border:1px solid rgba(99, 102, 241, 0.35); color:#A5B4FC; font-size:0.68rem; font-weight:700; letter-spacing:0.08em; text-transform:uppercase; padding:0.25rem 0.65rem; border-radius:6px;">
                        <i class="fa-solid fa-graduation-cap" style="margin-right:0.35rem;"></i> Data Source 2 &middot; Zenodo
                    </span>
                    <span style="color:#94A3B8; font-size:0.75rem; font-weight:600;">Manik Sonawane</span>
                </div>
                <div style="color:#F1F5F9; font-size:1.02rem; font-weight:700; margin-bottom:0.35rem;">
                    Artificial Consumer Shopping Behavior Data for Marketing and Analytics
                </div>
                <div style="color:#94A3B8; font-size:0.83rem; line-height:1.5; margin-bottom:1rem;">
                    Open-access research dataset (10,000 synthetic records) curated for marketing research, consumer segmentation, predictive classification, and retail analytics without PII concerns.
                </div>
            </div>
            <div>
                <a href="https://zenodo.org/records/17036147" target="_blank" style="
                    display: inline-flex;
                    align-items: center;
                    justify-content: center;
                    gap: 0.5rem;
                    width: 100%;
                    background: rgba(99, 102, 241, 0.12);
                    border: 1px solid rgba(99, 102, 241, 0.35);
                    color: #FFFFFF !important;
                    -webkit-text-fill-color: #FFFFFF !important;
                    border-radius: 8px;
                    padding: 0.6rem 1rem;
                    font-size: 0.82rem;
                    font-weight: 600;
                    text-decoration: none;
                    transition: all 0.2s ease;
                ">
                    <i class="fa-solid fa-atom" style="color: #A5B4FC;"></i>
                    <span>Open Zenodo Record (DOI: 10.5281/zenodo.17036147)</span>
                    <i class="fa-solid fa-arrow-up-right-from-square" style="font-size: 0.65rem; color: #94A3B8;"></i>
                </a>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # 2 & 3. Before Cleaning (Raw) & After Cleaning (Processed) Dataset Cards
    raw_path = "customer_data.csv"
    clean_path = "clean_customer_data.csv"
    raw_exists = os.path.exists(raw_path)
    clean_exists = os.path.exists(clean_path)

    raw_bytes = None
    if raw_exists:
        with open(raw_path, "rb") as f:
            raw_bytes = f.read()

    clean_bytes = None
    if clean_exists:
        with open(clean_path, "rb") as f:
            clean_bytes = f.read()

    c_raw, c_clean = st.columns(2, gap="medium")

    with c_raw:
        st.markdown(f"""
        <div style="
            background: #111827;
            border: 1px solid rgba(245, 158, 11, 0.28);
            border-radius: 12px;
            padding: 1.25rem 1.4rem;
            margin-bottom: 0.8rem;
        ">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.75rem;">
                <span style="background:rgba(245, 158, 11, 0.15); border:1px solid rgba(245, 158, 11, 0.35); color:#FBBF24; font-size:0.68rem; font-weight:700; letter-spacing:0.08em; text-transform:uppercase; padding:0.25rem 0.65rem; border-radius:6px;">
                    <i class="fa-solid fa-file-csv" style="margin-right:0.35rem;"></i> Before Cleaning &middot; Raw Dataset
                </span>
                <span style="color:#94A3B8; font-size:0.76rem; font-family:monospace; background:rgba(255,255,255,0.05); padding:0.2rem 0.5rem; border-radius:4px;">customer_data.csv</span>
            </div>
            <div style="color:#F1F5F9; font-size:1.02rem; font-weight:700; margin-bottom:0.35rem;">
                Original Customer Behaviour Dataset
            </div>
            <div style="color:#94A3B8; font-size:0.83rem; line-height:1.5;">
                <strong>Purpose:</strong> Stores the original, unaltered consumer records directly as received. Contains human-readable text categories and baseline metrics for authentic business auditing.
            </div>
        </div>
        """, unsafe_allow_html=True)

        if raw_exists and raw_bytes is not None:
            st.download_button(
                label="📥  Download Raw Dataset (customer_data.csv)",
                data=raw_bytes,
                file_name="customer_data.csv",
                mime="text/csv",
                key="btn_download_raw_csv",
                use_container_width=True
            )
        else:
            st.warning("⚠️ customer_data.csv is not available for download.")

    with c_clean:
        st.markdown(f"""
        <div style="
            background: #111827;
            border: 1px solid rgba(16, 185, 129, 0.28);
            border-radius: 12px;
            padding: 1.25rem 1.4rem;
            margin-bottom: 0.8rem;
        ">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.75rem;">
                <span style="background:rgba(16, 185, 129, 0.15); border:1px solid rgba(16, 185, 129, 0.35); color:#34D399; font-size:0.68rem; font-weight:700; letter-spacing:0.08em; text-transform:uppercase; padding:0.25rem 0.65rem; border-radius:6px;">
                    <i class="fa-solid fa-wand-magic-sparkles" style="margin-right:0.35rem;"></i> After Cleaning &middot; Processed Dataset
                </span>
                <span style="color:#94A3B8; font-size:0.76rem; font-family:monospace; background:rgba(255,255,255,0.05); padding:0.2rem 0.5rem; border-radius:4px;">clean_customer_data.csv</span>
            </div>
            <div style="color:#F1F5F9; font-size:1.02rem; font-weight:700; margin-bottom:0.35rem;">
                Cleaned, Transformed &amp; Encoded Dataset
            </div>
            <div style="color:#94A3B8; font-size:0.83rem; line-height:1.5;">
                <strong>Purpose:</strong> Cleaned, transformed, and feature-engineered dataset formatted specifically for statistical analytics, Random Forest inference, and K-Means customer segmentation.
            </div>
        </div>
        """, unsafe_allow_html=True)

        if clean_exists and clean_bytes is not None:
            st.download_button(
                label="📥  Download Cleaned Dataset (clean_customer_data.csv)",
                data=clean_bytes,
                file_name="clean_customer_data.csv",
                mime="text/csv",
                key="btn_download_clean_csv",
                use_container_width=True
            )
        else:
            st.warning("⚠️ clean_customer_data.csv is not available for download.")

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
    (
        t1,
        "fa-desktop",
        "Frontend",
        [
            ("Streamlit", "Interactive web dashboard"),
            ("Plotly", "Interactive charts and visualizations"),
            ("Custom CSS", "Dark theme and responsive styling"),
        ],
    ),
    (
        t2,
        "fa-table-cells",
        "Data Processing & Analytics",
        [
            ("Pandas & NumPy", "Data cleaning & numerical computations"),
            ("Data Preprocessing", "Missing values & categorical encoding"),
            ("Exploratory Data Analysis (EDA)", "Behavioral & pattern discovery"),
        ],
    ),
    (
        t3,
        "fa-robot",
        "Machine Learning",
        [
            ("Scikit-learn", "Model development"),
            ("Random Forest", "Customer satisfaction prediction"),
            ("K-Means", "Customer segmentation (4 personas)"),
        ],
    ),
    (
        t4,
        "fa-cloud",
        "Deployment & Model Management",
        [
            ("Streamlit Cloud", "Deployment support"),
            ("Joblib", "Model persistence"),
            ("Real-time Prediction", "Model inference through the dashboard"),
        ],
    ),
]
for col, icon, title, items in stack:
    rows = "".join(
        f'<div style="color:#94A3B8;font-size:0.79rem;padding:0.32rem 0;display:flex;align-items:flex-start;gap:0.45rem;line-height:1.4;">'
        f'<i class="fa-solid fa-check" style="font-size:0.62rem;color:#6366F1;margin-top:0.3rem;flex-shrink:0;"></i>'
        f'<div><strong style="color:#F1F5F9;font-weight:600;">{name}</strong> &mdash; <span>{desc}</span></div>'
        f'</div>'
        for name, desc in items
    )
    with col:
        st.markdown(f"""
        <div class="feature-card" style="padding:1.25rem 1.3rem;display:flex;flex-direction:column;justify-content:flex-start;height:100%;">
            <div class="fc-icon-wrap"><i class="fa-solid {icon}"></i></div>
            <div class="fc-title" style="font-size:0.95rem;font-weight:700;margin-bottom:0.65rem;color:#FFFFFF;min-height:2.4rem;display:flex;align-items:center;">{title}</div>
            <div style="display:flex;flex-direction:column;gap:0.2rem;">
                {rows}
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="
    margin-top:2rem;padding:1.6rem 2rem;
    background:linear-gradient(135deg,#111827 0%,#1A2234 100%);
    border:1px solid #1F2D45;border-radius:14px;text-align:center;
">
    <div style="color:#F1F5F9;font-size:1.05rem;font-weight:700;margin-bottom:0.4rem;letter-spacing:-0.01em;">
        <i class="fa-solid fa-chart-pie" style="color:#6366F1;margin-right:0.45rem;"></i>
        Customer Behavior Analytics Platform
    </div>
    <div style="color:#94A3B8;font-size:0.83rem;line-height:1.6;">
        Data-Driven Insights &middot; Customer Satisfaction Prediction &middot; Customer Segmentation &middot; 10,000 Customer Records
    </div>
    <div style="margin-top:0.85rem;display:flex;gap:0.6rem;justify-content:center;flex-wrap:wrap;">
        <span class="badge badge-indigo"><i class="fa-solid fa-brain"></i> Machine Learning</span>
        <span class="badge badge-amber"><i class="fa-solid fa-chart-line"></i> Interactive Analytics</span>
        <span class="badge badge-green"><i class="fa-solid fa-users"></i> Customer Insights</span>
    </div>
</div>
""", unsafe_allow_html=True)
