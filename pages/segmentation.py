"""
Customer Segmentation — 10k Consumer Behaviour Dataset
Behavioral personas discovered dynamically from actual cluster profiles using KMeans clustering.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils import (
    load_data, load_segmentation_model, render_sidebar,
    get_chart_layout, get_cluster_labels, FA_CDN,
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

# ── Assign clusters dynamically to dataset ────────────────────────────────────
Xs = df[SEG_FEATURE_COLS].copy()
df = df.copy()
df["Cluster"] = kmeans.predict(scaler.transform(Xs))
CLUSTER_LABELS = get_cluster_labels(df)
df["Segment"] = df["Cluster"].map(CLUSTER_LABELS)

CLUSTER_COLORS = ["#6366F1", "#10B981", "#F59E0B", "#EF4444"]
SEGMENT_COLOR_MAP = {
    CLUSTER_LABELS[cid]: CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
    for i, cid in enumerate(sorted(CLUSTER_LABELS.keys()))
}

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown(f"""
{FA_CDN}
<div class="hero-banner" style="padding:1.6rem 2rem;">
    <div class="hero-tag"><i class="fa-solid fa-layer-group"></i> KMeans Clustering</div>
    <h1>Customer Segmentation</h1>
    <p class="hero-sub">Discover behavioral personas derived dynamically from 10,000 customer records using KMeans clustering across demographics, spending, income tier, category preferences, and loyalty membership.</p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── Cluster summary cards ─────────────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="section-accent"></div>
    <h2>Segment Overview</h2>
</div>""", unsafe_allow_html=True)

cols = st.columns(len(CLUSTER_LABELS), gap="small")
for i, cid in enumerate(sorted(CLUSTER_LABELS.keys())):
    label = CLUSTER_LABELS[cid]
    seg_df = df[df["Cluster"] == cid]
    count = len(seg_df)
    pct = count / len(df) * 100
    avg_spd = seg_df["Avg_Spend"].mean()
    avg_sat = seg_df["Satisfaction_Score"].mean()
    loyalty_col = "Loyalty_Member" if "Loyalty_Member" in seg_df.columns else "Membership_Status"
    loyalty = (seg_df[loyalty_col] == "Yes").mean() * 100 if loyalty_col in seg_df.columns else 0.0
    clr = SEGMENT_COLOR_MAP[label]

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
                    <div class="sc-stat-value">{loyalty:.1f}%</div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

st.divider()

# ── Cluster Visualisation ─────────────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="section-accent"></div>
    <h2>Cluster Visualisation</h2>
</div>""", unsafe_allow_html=True)

sample_df = df.sample(min(3000, len(df)), random_state=42)
cv1, cv2 = st.columns(2, gap="small")
with cv1:
    fig_sc = px.scatter(
        sample_df,
        x="Avg_Spend", y="Satisfaction_Score",
        color="Segment", color_discrete_map=SEGMENT_COLOR_MAP,
        category_orders={"Segment": [CLUSTER_LABELS[c] for c in sorted(CLUSTER_LABELS.keys())]},
        opacity=0.65, hover_data=["Age", "Income_Level", "Region"],
        labels={"Avg_Spend": "Avg Spend ($)", "Satisfaction_Score": "Satisfaction Score"}
    )
    fig_sc.update_layout(**get_chart_layout(height=420, title="Spend vs Satisfaction by Segment"))
    st.plotly_chart(fig_sc, use_container_width=True)

with cv2:
    fig_sc2 = px.scatter(
        sample_df,
        x="Age", y="Avg_Spend",
        color="Segment", color_discrete_map=SEGMENT_COLOR_MAP,
        category_orders={"Segment": [CLUSTER_LABELS[c] for c in sorted(CLUSTER_LABELS.keys())]},
        opacity=0.65, hover_data=["Income_Level", "Purchase_Category"],
        labels={"Age": "Age (years)", "Avg_Spend": "Avg Spend ($)"}
    )
    fig_sc2.update_layout(**get_chart_layout(height=420, title="Age vs Spend by Segment"))
    st.plotly_chart(fig_sc2, use_container_width=True)

st.divider()

# ── Segment Behaviour Breakdown ───────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="section-accent"></div>
    <h2>Segment Behaviour Breakdown</h2>
</div>""", unsafe_allow_html=True)

sb1, sb2 = st.columns(2, gap="small")
with sb1:
    seg_cat = df.groupby(["Segment", "Purchase_Category"]).size().reset_index(name="Count")
    fig_scat = px.bar(
        seg_cat, x="Segment", y="Count", color="Purchase_Category",
        barmode="group",
        color_discrete_sequence=["#6366F1","#10B981","#F59E0B","#EF4444","#8B5CF6","#06B6D4"],
        category_orders={"Segment": [CLUSTER_LABELS[c] for c in sorted(CLUSTER_LABELS.keys())]},
        labels={"Count": "Customers", "Purchase_Category": "Category"}
    )
    fig_scat.update_layout(**get_chart_layout(height=380, show_legend=True, title="Category Preference by Segment"))
    st.plotly_chart(fig_scat, use_container_width=True)

with sb2:
    seg_income = df.groupby(["Segment", "Income_Level"]).size().reset_index(name="Count")
    fig_sinc = px.bar(
        seg_income, x="Segment", y="Count", color="Income_Level",
        barmode="stack",
        color_discrete_map={"High":"#10B981","Medium":"#6366F1","Low":"#EF4444"},
        category_orders={
            "Segment": [CLUSTER_LABELS[c] for c in sorted(CLUSTER_LABELS.keys())],
            "Income_Level": ["Low", "Medium", "High"]
        },
        labels={"Count": "Customers", "Income_Level": "Income Tier"}
    )
    fig_sinc.update_layout(**get_chart_layout(height=380, show_legend=True, title="Income Mix by Segment"))
    st.plotly_chart(fig_sinc, use_container_width=True)

st.divider()

# ── Cluster Comparison ────────────────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="section-accent"></div>
    <h2>Cluster Comparison</h2>
</div>""", unsafe_allow_html=True)

comp_data = []
loyalty_col = "Loyalty_Member" if "Loyalty_Member" in df.columns else "Membership_Status"

for cid in sorted(CLUSTER_LABELS.keys()):
    label = CLUSTER_LABELS[cid]
    sub = df[df["Cluster"] == cid]
    c_count = len(sub)
    c_pct = (c_count / len(df)) * 100
    c_spend = sub["Avg_Spend"].mean()
    c_sat = sub["Satisfaction_Score"].mean()
    c_loyalty = (sub[loyalty_col] == "Yes").mean() * 100 if loyalty_col in sub.columns else 0.0

    comp_data.append({
        "Cluster": f"C{cid}: {label}",
        "Segment": label,
        "Customer Count": c_count,
        "Percentage": c_pct,
        "Avg Spend ($)": c_spend,
        "Avg Satisfaction": c_sat,
        "Membership Rate (%)": c_loyalty
    })

comp_df = pd.DataFrame(comp_data)

cc1, cc2 = st.columns(2, gap="small")

with cc1:
    fig_comp_count = px.bar(
        comp_df,
        x="Segment",
        y="Customer Count",
        color="Segment",
        color_discrete_map=SEGMENT_COLOR_MAP,
        text=[f"{row['Customer Count']:,} ({row['Percentage']:.1f}%)" for _, row in comp_df.iterrows()],
        category_orders={"Segment": [CLUSTER_LABELS[c] for c in sorted(CLUSTER_LABELS.keys())]},
        labels={"Customer Count": "Total Customers"}
    )
    fig_comp_count.update_traces(textposition="outside", width=0.45)
    lay_cc = get_chart_layout(height=360, show_legend=False, title="Customer Count & Share across Clusters")
    lay_cc["yaxis"] = dict(gridcolor="#1F2D45", range=[0, comp_df["Customer Count"].max() * 1.18])
    fig_comp_count.update_layout(**lay_cc)
    st.plotly_chart(fig_comp_count, use_container_width=True)

with cc2:
    fig_comp_spend = px.bar(
        comp_df,
        x="Segment",
        y="Avg Spend ($)",
        color="Segment",
        color_discrete_map=SEGMENT_COLOR_MAP,
        text=[f"${row['Avg Spend ($)']:.2f}" for _, row in comp_df.iterrows()],
        category_orders={"Segment": [CLUSTER_LABELS[c] for c in sorted(CLUSTER_LABELS.keys())]},
        labels={"Avg Spend ($)": "Average Spend ($)"}
    )
    fig_comp_spend.update_traces(textposition="outside", width=0.45)
    lay_cs = get_chart_layout(height=360, show_legend=False, title="Average Spend ($) across Clusters")
    lay_cs["yaxis"] = dict(gridcolor="#1F2D45", range=[0, comp_df["Avg Spend ($)"].max() * 1.18])
    fig_comp_spend.update_layout(**lay_cs)
    st.plotly_chart(fig_comp_spend, use_container_width=True)

cc3, cc4 = st.columns(2, gap="small")

with cc3:
    fig_comp_sat = px.bar(
        comp_df,
        x="Segment",
        y="Avg Satisfaction",
        color="Segment",
        color_discrete_map=SEGMENT_COLOR_MAP,
        text=[f"{row['Avg Satisfaction']:.2f} / 5.0" for _, row in comp_df.iterrows()],
        category_orders={"Segment": [CLUSTER_LABELS[c] for c in sorted(CLUSTER_LABELS.keys())]},
        labels={"Avg Satisfaction": "Average Score (1–5)"}
    )
    fig_comp_sat.update_traces(textposition="outside", width=0.45)
    lay_sat = get_chart_layout(height=360, show_legend=False, title="Average Satisfaction Score across Clusters")
    lay_sat["yaxis"] = dict(gridcolor="#1F2D45", range=[0, 5.2])
    fig_comp_sat.update_layout(**lay_sat)
    st.plotly_chart(fig_comp_sat, use_container_width=True)

with cc4:
    fig_comp_loyalty = px.bar(
        comp_df,
        x="Segment",
        y="Membership Rate (%)",
        color="Segment",
        color_discrete_map=SEGMENT_COLOR_MAP,
        text=[f"{row['Membership Rate (%)']:.1f}%" for _, row in comp_df.iterrows()],
        category_orders={"Segment": [CLUSTER_LABELS[c] for c in sorted(CLUSTER_LABELS.keys())]},
        labels={"Membership Rate (%)": "Loyalty Rate (%)"}
    )
    fig_comp_loyalty.update_traces(textposition="outside", width=0.45)
    lay_loy = get_chart_layout(height=360, show_legend=False, title="Membership Rate (%) across Clusters")
    lay_loy["yaxis"] = dict(gridcolor="#1F2D45", range=[0, 118])
    fig_comp_loyalty.update_layout(**lay_loy)
    st.plotly_chart(fig_comp_loyalty, use_container_width=True)

# ── Summary Data Table ────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top: 1.2rem; margin-bottom: 0.6rem; color:#94A3B8; font-size:0.85rem; font-weight:600;">
    <i class="fa-solid fa-table-list" style="color:#6366F1; margin-right:0.35rem;"></i> Cluster Comparison Summary Table
</div>
""", unsafe_allow_html=True)

styled_df = comp_df.copy()
styled_df["Customers"] = styled_df.apply(lambda r: f"{r['Customer Count']:,} ({r['Percentage']:.1f}%)", axis=1)
styled_df["Avg Spend"] = styled_df["Avg Spend ($)"].apply(lambda v: f"${v:.2f}")
styled_df["Avg Satisfaction"] = styled_df["Avg Satisfaction"].apply(lambda v: f"{v:.2f} / 5.0")
styled_df["Membership Rate"] = styled_df["Membership Rate (%)"].apply(lambda v: f"{v:.1f}%")

display_cols = ["Cluster", "Customers", "Avg Spend", "Avg Satisfaction", "Membership Rate"]
st.dataframe(styled_df[display_cols], use_container_width=True, hide_index=True)

st.markdown("""
<div style="text-align:center;padding:1.4rem 0 0.5rem 0;color:#2D3B55;font-size:0.75rem;">
    Customer Segmentation • 10,000 Records • Dynamic KMeans Personas
</div>
""", unsafe_allow_html=True)
