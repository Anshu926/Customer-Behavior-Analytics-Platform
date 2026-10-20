"""
Customer Segmentation — 10k Consumer Behaviour Dataset
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils import (
    load_data, load_segmentation_model, render_sidebar,
    get_chart_layout, segment_customer, get_cluster_labels,
    INCOME_MAPPING, CATEGORY_MAPPING, MEMBERSHIP_MAPPING, FA_CDN,
    SEG_FEATURE_COLS
)

st.set_page_config(page_title="Customer Segmentation", page_icon="🔵", layout="wide", initial_sidebar_state="expanded")
render_sidebar("segmentation")

@st.cache_resource
def get_models():
    df = load_data()
    km, sc = load_segmentation_model(df)
    return km, sc, df

kmeans, scaler, df = get_models()

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

# ── Classify a new customer ───────────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="section-accent"></div>
    <h2>Classify a Customer into a Persona</h2>
</div>""", unsafe_allow_html=True)

nc1, nc2, nc3 = st.columns(3, gap="medium")
with nc1:
    n_age      = st.slider("Customer Age", 18, 69, 35)
    n_income   = st.selectbox("Income Bracket", ["Low", "Medium", "High"], index=1)

with nc2:
    n_spend    = st.slider("Average Spend ($)", 10.0, 630.0, 210.0, step=5.0)
    n_items    = st.slider("Items Purchased (Basket)", 1, 20, 5)

with nc3:
    n_recency  = st.slider("Days Since Last Purchase", 1, 90, 20)
    n_loyalty  = st.selectbox("Membership Status", ["No", "Yes"], index=1)

if st.button("🔍  Classify Customer Persona", use_container_width=True):
    seg = segment_customer(kmeans, scaler, {
        "age": n_age, "income": n_income,
        "avg_spend": n_spend, "items_purchased": n_items,
        "days_since_last_purchase": n_recency, "membership": n_loyalty
    })
    idx   = list(CLUSTER_LABELS.values()).index(seg)
    color = CLUSTER_COLORS[idx]

    strategies = {
        "Engaged Members": "Offer loyalty-tier points multipliers, early access sales, and referral bonuses.",
        "Prospect Shoppers": "Launch a targeted onboarding campaign with welcome discounts to convert into enrolled members.",
        "At-Risk Inactive": "Deploy high-priority re-engagement incentives, personalized win-back emails, and cart revival offers.",
        "High-Value Champions": "Deliver VIP concierge treatment, exclusive previews, luxury product bundles, and anniversary gifts.",
    }

    st.markdown(f"""
    <div style="background:{color}18;border:1px solid {color}55;border-radius:12px;
                padding:1.5rem;text-align:center;margin-top:0.8rem;">
        <div style="color:#94A3B8;font-size:0.72rem;font-weight:700;letter-spacing:0.15em;text-transform:uppercase;margin-bottom:0.5rem;">
            Assigned Behavioral Persona
        </div>
        <div style="color:{color};font-size:1.8rem;font-weight:700;margin-bottom:0.6rem;">{seg}</div>
        <div style="color:#CBD5E1;font-size:0.88rem;max-width:650px;margin:0 auto;line-height:1.6;">
            <strong>Targeted Strategy:</strong> {strategies.get(seg, '')}
        </div>
    </div>""", unsafe_allow_html=True)
