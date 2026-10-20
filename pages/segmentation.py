"""
Customer Segmentation — 10k Consumer Behaviour Dataset
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils import (
    load_data, load_segmentation_model, train_segmentation_model, ensure_features, render_sidebar,
    get_chart_layout, get_cluster_labels,
    INCOME_MAPPING, CATEGORY_MAPPING, MEMBERSHIP_MAPPING, FA_CDN,
    SEG_FEATURE_COLS
)

st.set_page_config(page_title="Customer Segmentation", page_icon="🔵", layout="wide", initial_sidebar_state="expanded")
render_sidebar("segmentation")

@st.cache_resource(ttl=3600)
def load_segmentation_pipeline(v="v10k_prod"):
    df = load_data()
    df = ensure_features(df)
    km, sc = load_segmentation_model(df)
    return km, sc, df

kmeans, scaler, df = load_segmentation_pipeline()
df = ensure_features(df)

CLUSTER_COLORS = ["#6366F1", "#10B981", "#EF4444", "#F59E0B"]
CLUSTER_LABELS = get_cluster_labels()

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown(f"""
{FA_CDN}
<div class="hero-banner" style="padding:1.6rem 2rem;">
    <div class="hero-tag"><i class="fa-solid fa-layer-group"></i> KMeans Clustering</div>
    <h1>Customer Segmentation</h1>
    <p class="hero-sub">Discover customer personas from 10,000 records using KMeans clustering across monetary spend, basket size, purchase recency, income level, and loyalty membership.</p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── Assign clusters to dataset ───────────────────────────────────────────────
Xs = df[SEG_FEATURE_COLS].copy()
df = df.copy()
try:
    df["Cluster"] = kmeans.predict(scaler.transform(Xs))
except Exception:
    kmeans, scaler = train_segmentation_model(df)
    df["Cluster"] = kmeans.predict(scaler.transform(Xs))
df["Segment"] = df["Cluster"].map(CLUSTER_LABELS)

# ── Cluster summary cards ─────────────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="section-accent"></div>
    <h2>Customer Personas Overview</h2>
</div>""", unsafe_allow_html=True)

cols = st.columns(4, gap="small")
for i, (cid, label) in enumerate(CLUSTER_LABELS.items()):
    seg_df  = df[df["Cluster"] == cid]
    count   = len(seg_df)
    pct     = count / len(df) * 100
    avg_spd = seg_df["Avg_Spend"].mean()
    avg_sat = seg_df["Satisfaction_Score"].mean()
    avg_itm = seg_df["Items_Purchased"].mean() if "Items_Purchased" in seg_df.columns else 0.0
    avg_rec = seg_df["Days_Since_Last_Purchase"].mean() if "Days_Since_Last_Purchase" in seg_df.columns else 0.0
    mem_col = "Membership_Status" if "Membership_Status" in seg_df.columns else "Loyalty_Member"
    loyalty = (seg_df[mem_col] == "Yes").mean() * 100
    clr     = CLUSTER_COLORS[i]

    # Persona icon
    icons = {0: "fa-gem", 1: "fa-bullseye", 2: "fa-triangle-exclamation", 3: "fa-crown"}
    icon  = icons.get(cid, "fa-users")

    with cols[i]:
        st.markdown(f"""
        <div class="seg-card">
            <div class="sc-header">
                <div class="sc-stripe" style="background:{clr};"></div>
                <div class="sc-icon-wrap" style="background:{clr}22;color:{clr};">
                    <i class="fa-solid {icon}"></i>
                </div>
                <div class="sc-title">{label}</div>
            </div>
            <div class="sc-stats">
                <div>
                    <div class="sc-stat-label">Customers</div>
                    <div class="sc-stat-value">{count:,} ({pct:.1f}%)</div>
                </div>
                <div>
                    <div class="sc-stat-label">Avg Spend</div>
                    <div class="sc-stat-value">${avg_spd:.0f}</div>
                </div>
                <div>
                    <div class="sc-stat-label">Avg Basket</div>
                    <div class="sc-stat-value">{avg_itm:.1f} items</div>
                </div>
                <div>
                    <div class="sc-stat-label">Avg Recency</div>
                    <div class="sc-stat-value">{avg_rec:.0f} days</div>
                </div>
                <div>
                    <div class="sc-stat-label">Satisfaction</div>
                    <div class="sc-stat-value">{avg_sat:.2f} / 5</div>
                </div>
                <div>
                    <div class="sc-stat-label">Loyalty %</div>
                    <div class="sc-stat-value">{loyalty:.1f}%</div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

st.divider()

# ── Cluster scatter plots ─────────────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="section-accent"></div>
    <h2>Multi-Dimensional Persona Visualisations</h2>
</div>""", unsafe_allow_html=True)

cv1, cv2 = st.columns(2, gap="small")
sample_df = df.sample(min(3000, len(df)), random_state=42)

with cv1:
    fig_sc = px.scatter(sample_df,
                        x="Avg_Spend", y="Days_Since_Last_Purchase",
                        color="Segment", color_discrete_map={CLUSTER_LABELS[i]: CLUSTER_COLORS[i] for i in range(4)},
                        opacity=0.6, hover_data=["Age", "Income_Level", "Items_Purchased"],
                        labels={"Avg_Spend": "Average Spend ($)", "Days_Since_Last_Purchase": "Days Since Last Purchase (Recency)"})
    fig_sc.update_layout(**get_chart_layout(height=420, title="Spend vs Recency (Days) by Persona"))
    st.plotly_chart(fig_sc, use_container_width=True)

with cv2:
    fig_sc2 = px.scatter(sample_df,
                         x="Items_Purchased", y="Avg_Spend",
                         color="Segment", color_discrete_map={CLUSTER_LABELS[i]: CLUSTER_COLORS[i] for i in range(4)},
                         opacity=0.6, hover_data=["Age", "Income_Level", "Satisfaction_Score"],
                         labels={"Items_Purchased": "Basket Size (Items)", "Avg_Spend": "Average Spend ($)"})
    fig_sc2.update_layout(**get_chart_layout(height=420, title="Basket Size vs Spend ($) by Persona"))
    st.plotly_chart(fig_sc2, use_container_width=True)

st.divider()

# ── Category & Income mix per segment ────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="section-accent"></div>
    <h2>Behavioral &amp; Demographic Breakdown by Segment</h2>
</div>""", unsafe_allow_html=True)

sb1, sb2 = st.columns(2, gap="small")
with sb1:
    seg_cat = df.groupby(["Segment", "Purchase_Category"]).size().reset_index(name="Count")
    fig_scat = px.bar(seg_cat, x="Segment", y="Count", color="Purchase_Category",
                      barmode="group", color_discrete_sequence=["#6366F1","#10B981","#F59E0B","#EF4444","#8B5CF6","#06B6D4"])
    fig_scat.update_layout(**get_chart_layout(height=380, show_legend=True, title="Category Preferences across Segments"))
    st.plotly_chart(fig_scat, use_container_width=True)

with sb2:
    seg_income = df.groupby(["Segment", "Income_Level"]).size().reset_index(name="Count")
    fig_sinc = px.bar(seg_income, x="Segment", y="Count", color="Income_Level",
                      barmode="stack", color_discrete_map={"High":"#10B981","Medium":"#6366F1","Low":"#EF4444"})
    fig_sinc.update_layout(**get_chart_layout(height=380, show_legend=True, title="Income Distribution across Segments"))
    st.plotly_chart(fig_sinc, use_container_width=True)

st.divider()

# ── Segment Insights ──────────────────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="section-accent"></div>
    <h2>Segment Insights</h2>
</div>""", unsafe_allow_html=True)

# 1. Dataset-driven Key Segment Insights
seg_counts = df["Segment"].value_counts()
largest_seg = seg_counts.idxmax()
largest_count = int(seg_counts.max())
largest_pct = (largest_count / len(df)) * 100

seg_spend = df.groupby("Segment")["Avg_Spend"].mean()
highest_spend_seg = seg_spend.idxmax()
highest_spend_val = float(seg_spend.max())

rec_col = "Days_Since_Last_Purchase" if "Days_Since_Last_Purchase" in df.columns else "Recency"
seg_recency = df.groupby("Segment")[rec_col].mean()
most_inactive_seg = seg_recency.idxmax()
most_inactive_days = float(seg_recency.max())

prospect_seg = "Prospect Shoppers"
prospect_count = int(seg_counts.get(prospect_seg, (df["Segment"] == prospect_seg).sum()))
prospect_pct = (prospect_count / len(df)) * 100 if len(df) > 0 else 0.0

i1, i2, i3, i4 = st.columns(4, gap="small")

with i1:
    st.markdown(f"""
    <div class="result-card" style="background:rgba(99,102,241,0.08);border:1px solid rgba(99,102,241,0.28);padding:1.25rem 1.2rem;height:100%;display:flex;flex-direction:column;justify-content:space-between;">
        <div>
            <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:0.5rem;">
                <div class="rc-label" style="font-size:0.68rem;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:#A5B4FC;">
                    Largest Customer Segment
                </div>
                <i class="fa-solid fa-users" style="color:#818CF8;font-size:1.05rem;"></i>
            </div>
            <div style="color:#F1F5F9;font-size:1.18rem;font-weight:700;margin-bottom:0.3rem;">
                {largest_seg}
            </div>
            <div style="color:#818CF8;font-size:0.88rem;font-weight:600;">
                {largest_count:,} Customers <span style="color:#94A3B8;font-weight:400;font-size:0.78rem;">({largest_pct:.1f}% of total)</span>
            </div>
        </div>
        <div style="color:#94A3B8;font-size:0.76rem;margin-top:0.45rem;line-height:1.45;">
            Forms the core volume of active shoppers with consistent purchase habits and high engagement.
        </div>
    </div>""", unsafe_allow_html=True)

with i2:
    st.markdown(f"""
    <div class="result-card" style="background:rgba(245,158,11,0.08);border:1px solid rgba(245,158,11,0.28);padding:1.25rem 1.2rem;height:100%;display:flex;flex-direction:column;justify-content:space-between;">
        <div>
            <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:0.5rem;">
                <div class="rc-label" style="font-size:0.68rem;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:#FBBF24;">
                    Highest Avg-Spending Segment
                </div>
                <i class="fa-solid fa-crown" style="color:#F59E0B;font-size:1.05rem;"></i>
            </div>
            <div style="color:#F1F5F9;font-size:1.18rem;font-weight:700;margin-bottom:0.3rem;">
                {highest_spend_seg}
            </div>
            <div style="color:#FBBF24;font-size:0.88rem;font-weight:600;">
                ${highest_spend_val:.2f} <span style="color:#94A3B8;font-weight:400;font-size:0.78rem;">Average Spend</span>
            </div>
        </div>
        <div style="color:#94A3B8;font-size:0.76rem;margin-top:0.45rem;line-height:1.45;">
            Drives top-tier basket values and revenue yield with the highest customer satisfaction.
        </div>
    </div>""", unsafe_allow_html=True)

with i3:
    st.markdown(f"""
    <div class="result-card" style="background:rgba(239,68,68,0.08);border:1px solid rgba(239,68,68,0.28);padding:1.25rem 1.2rem;height:100%;display:flex;flex-direction:column;justify-content:space-between;">
        <div>
            <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:0.5rem;">
                <div class="rc-label" style="font-size:0.68rem;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:#F87171;">
                    Most Inactive Segment
                </div>
                <i class="fa-solid fa-triangle-exclamation" style="color:#EF4444;font-size:1.05rem;"></i>
            </div>
            <div style="color:#F1F5F9;font-size:1.18rem;font-weight:700;margin-bottom:0.3rem;">
                {most_inactive_seg}
            </div>
            <div style="color:#F87171;font-size:0.88rem;font-weight:600;">
                {most_inactive_days:.1f} Days <span style="color:#94A3B8;font-weight:400;font-size:0.78rem;">Average Inactivity</span>
            </div>
        </div>
        <div style="color:#94A3B8;font-size:0.76rem;margin-top:0.45rem;line-height:1.45;">
            Exhibits extended dormancy and lower review ratings; primary priority for churn prevention.
        </div>
    </div>""", unsafe_allow_html=True)

with i4:
    st.markdown(f"""
    <div class="result-card" style="background:rgba(16,185,129,0.08);border:1px solid rgba(16,185,129,0.28);padding:1.25rem 1.2rem;height:100%;display:flex;flex-direction:column;justify-content:space-between;">
        <div>
            <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:0.5rem;">
                <div class="rc-label" style="font-size:0.68rem;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:#34D399;">
                    Prospect Shoppers &mdash; Conversion Opportunity
                </div>
                <i class="fa-solid fa-bullseye" style="color:#10B981;font-size:1.05rem;"></i>
            </div>
            <div style="color:#F1F5F9;font-size:1.18rem;font-weight:700;margin-bottom:0.3rem;">
                {prospect_seg}
            </div>
            <div style="color:#34D399;font-size:0.88rem;font-weight:600;">
                {prospect_count:,} Customers <span style="color:#94A3B8;font-weight:400;font-size:0.78rem;">({prospect_pct:.1f}% of total)</span>
            </div>
        </div>
        <div style="color:#94A3B8;font-size:0.76rem;margin-top:0.45rem;line-height:1.45;">
            Active frequent shoppers with zero loyalty enrollment; prime opportunity for membership conversion.
        </div>
    </div>""", unsafe_allow_html=True)
