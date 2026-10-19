"""
Customer Segmentation — KMeans Clustering
Segments customers using: Gender, Age, Income_Level, Region, Avg_Spend, Membership.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils import (
    load_data, load_segmentation_model, render_sidebar,
    get_chart_layout, segment_customer, get_cluster_labels,
    INCOME_MAPPING, GENDER_MAPPING, REGION_MAPPING, MEMBERSHIP_MAPPING,
    SEG_FEATURE_COLS, FA_CDN
)

st.set_page_config(page_title="Customer Segmentation", page_icon="🔵", layout="wide", initial_sidebar_state="expanded")
render_sidebar("segmentation")

@st.cache_resource
def get_models():
    df = load_data()
    km, sc = load_segmentation_model(df)
    return km, sc, df

kmeans, scaler, df = get_models()

CLUSTER_COLORS = ["#6366F1", "#10B981", "#F59E0B", "#EF4444"]
CLUSTER_LABELS = get_cluster_labels()

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown(f"""
{FA_CDN}
<div class="hero-banner" style="padding:1.6rem 2rem;">
    <div class="hero-tag"><i class="fa-solid fa-layer-group"></i> KMeans Clustering</div>
    <h1>Customer Segmentation</h1>
    <p class="hero-sub">Discover customer personas across 10,000 records using KMeans clustering across Age, Gender, Income, Region, Spend, and Membership status.</p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── Assign clusters to full dataset ──────────────────────────────────────────
Xs = df[SEG_FEATURE_COLS].copy()
df = df.copy()
df["Cluster"] = kmeans.predict(scaler.transform(Xs))
df["Segment"] = df["Cluster"].map(CLUSTER_LABELS)

# ── Cluster summary cards ─────────────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="section-accent"></div>
    <h2>Segment Overview</h2>
</div>""", unsafe_allow_html=True)

cols = st.columns(4, gap="small")
for i, (cid, label) in enumerate(CLUSTER_LABELS.items()):
    seg_df  = df[df["Cluster"] == cid]
    count   = len(seg_df)
    pct     = count / len(df) * 100
    avg_spd = seg_df["Avg_Spend"].mean()
    avg_sat = seg_df["Satisfaction_Score"].mean()
    mem_pct = (seg_df["Membership"] == "Yes").mean() * 100
    clr     = CLUSTER_COLORS[i]
    with cols[i]:
        st.markdown(f"""
        <div class="seg-card">
            <div class="sc-header">
                <div class="sc-stripe" style="background:{clr};"></div>
                <div class="sc-icon-wrap" style="background:{clr}22;color:{clr};">
                    <i class="fa-solid fa-users"></i>
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
                    <div class="sc-stat-label">Avg Satisfaction</div>
                    <div class="sc-stat-value">{avg_sat:.2f} / 5</div>
                </div>
                <div>
                    <div class="sc-stat-label">Membership Rate</div>
                    <div class="sc-stat-value">{mem_pct:.1f}%</div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

st.divider()

# ── Cluster scatter ───────────────────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="section-accent"></div>
    <h2>Cluster Visualisation</h2>
</div>""", unsafe_allow_html=True)

cv1, cv2 = st.columns(2, gap="small")
with cv1:
    fig_sc = px.scatter(df.sample(min(3000, len(df)), random_state=42),
                        x="Avg_Spend", y="Satisfaction_Score",
                        color="Segment", color_discrete_sequence=CLUSTER_COLORS,
                        opacity=0.6, hover_data=["Age", "Income_Level", "Region", "Membership"],
                        labels={"Avg_Spend": "Avg Spend ($)", "Satisfaction_Score": "Satisfaction Score"})
    fig_sc.update_layout(**get_chart_layout(height=420, title="Spend vs Satisfaction Score by Segment"))
    st.plotly_chart(fig_sc, use_container_width=True)

with cv2:
    fig_sc2 = px.scatter(df.sample(min(3000, len(df)), random_state=1),
                         x="Age", y="Avg_Spend",
                         color="Segment", color_discrete_sequence=CLUSTER_COLORS,
                         opacity=0.6, hover_data=["Income_Level", "Region", "Membership"],
                         labels={"Age": "Age (years)", "Avg_Spend": "Avg Spend ($)"})
    fig_sc2.update_layout(**get_chart_layout(height=420, title="Age vs Spend by Segment"))
    st.plotly_chart(fig_sc2, use_container_width=True)

st.divider()

# ── Income & Region mix per segment ──────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="section-accent"></div>
    <h2>Segment Composition Breakdown</h2>
</div>""", unsafe_allow_html=True)

sb1, sb2 = st.columns(2, gap="small")
with sb1:
    seg_income = df.groupby(["Segment", "Income_Level"]).size().reset_index(name="Count")
    fig_sinc = px.bar(seg_income, x="Segment", y="Count", color="Income_Level",
                      barmode="stack", color_discrete_map={"High":"#10B981","Medium":"#6366F1","Low":"#EF4444"})
    fig_sinc.update_layout(**get_chart_layout(height=380, show_legend=True, title="Income Tier Composition by Segment"))
    st.plotly_chart(fig_sinc, use_container_width=True)

with sb2:
    seg_reg = df.groupby(["Segment", "Region"]).size().reset_index(name="Count")
    fig_sreg = px.bar(seg_reg, x="Segment", y="Count", color="Region",
                      barmode="group", color_discrete_map={"Urban":"#6366F1","Semi-Urban":"#10B981","Rural":"#F59E0B"})
    fig_sreg.update_layout(**get_chart_layout(height=380, show_legend=True, title="Region Composition by Segment"))
    st.plotly_chart(fig_sreg, use_container_width=True)

st.divider()

# ── Classify a new customer ───────────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="section-accent"></div>
    <h2>Classify a New Customer Profile</h2>
</div>""", unsafe_allow_html=True)

nc1, nc2 = st.columns(2, gap="medium")
with nc1:
    n_age        = st.slider("Age", 18, 69, 32, key="seg_age")
    n_gender     = st.selectbox("Gender", ["F", "M", "Other"], key="seg_gender",
                                format_func=lambda x: {"F": "Female (F)", "M": "Male (M)", "Other": "Other"}.get(x, x))
    n_income     = st.selectbox("Income Level", ["Low", "Medium", "High"], index=1, key="seg_inc")
with nc2:
    n_region     = st.selectbox("Region", ["Rural", "Semi-Urban", "Urban"], index=2, key="seg_reg")
    n_spend      = st.slider("Avg Spend ($)", 10.0, 630.0, 200.0, step=5.0, key="seg_spd")
    n_membership = st.selectbox("Membership", ["Yes", "No"], index=0, key="seg_mem")

if st.button("🔍  Classify Customer Segment", use_container_width=True):
    seg = segment_customer(kmeans, scaler, {
        "age": n_age, "gender": n_gender, "income": n_income,
        "region": n_region, "avg_spend": n_spend, "membership": n_membership
    })
    idx   = list(CLUSTER_LABELS.values()).index(seg)
    color = CLUSTER_COLORS[idx]
    st.markdown(f"""
    <div style="background:{color}18;border:1px solid {color}55;border-radius:12px;
                padding:1.5rem;text-align:center;margin-top:0.5rem;">
        <div style="color:#94A3B8;font-size:0.72rem;font-weight:700;letter-spacing:0.15em;text-transform:uppercase;margin-bottom:0.5rem;">
            Assigned Segment Persona
        </div>
        <div style="color:{color};font-size:1.6rem;font-weight:700;">{seg}</div>
    </div>""", unsafe_allow_html=True)
