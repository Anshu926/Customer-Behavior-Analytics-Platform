"""
Business Dashboard — KPIs and visualizations (10k Consumer Behaviour Dataset)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils import load_data, get_data_stats, render_sidebar, get_chart_layout, FA_CDN

st.set_page_config(page_title="Business Dashboard", page_icon="📊", layout="wide", initial_sidebar_state="expanded")
render_sidebar("dashboard")

@st.cache_data
def get_cached_data():
    return load_data()

df    = get_cached_data()
stats = get_data_stats(df)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown(f"""
{FA_CDN}
<div class="hero-banner" style="padding:1.6rem 2rem;">
    <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:0.5rem;">
        <div class="hero-tag"><i class="fa-solid fa-chart-bar"></i> Real-time Analytics</div>
        <span style="display:inline-flex;align-items:center;gap:0.4rem;background:rgba(99,102,241,0.2);border:1px solid rgba(99,102,241,0.4);color:#FFFFFF;font-size:0.72rem;font-weight:700;padding:0.28rem 0.7rem;border-radius:8px;">
            <i class="fa-solid fa-database"></i> 10,000 Customers
        </span>
    </div>
    <h1>Business Dashboard</h1>
    <p class="hero-sub">Executive-level KPIs and comprehensive consumer behaviour insights — 10,000 synthetic customer records across 6 product categories, 3 regions, and 5 payment modes.</p>
</div>
""", unsafe_allow_html=True)

# ── KPI ribbon ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="section-accent"></div>
    <h2>Key Performance Indicators</h2>
</div>
""", unsafe_allow_html=True)

k1, k2, k3, k4, k5 = st.columns(5, gap="small")
kpis = [
    (k1, "fa-users",          "Total Customers",   f"{stats['total_customers']:,}",      "positive", "All records"),
    (k2, "fa-dollar-sign",    "Avg Spend",         f"${stats['avg_spend']:.2f}",          "positive", "Per customer"),
    (k3, "fa-star",           "Avg Satisfaction",  f"{stats['avg_score']:.2f} / 5.0",    "neutral",  "Overall score"),
    (k4, "fa-arrow-trend-up", "Total Revenue",     f"${stats['total_revenue']:,.0f}",     "positive", "Combined spend"),
    (k5, "fa-heart",          "Loyalty Members",   f"{stats['loyalty_pct']:.1f}%",        "positive", "Of all customers"),
]
for col, icon, label, value, cls, sub in kpis:
    with col:
        st.markdown(f"""
        <div class="pro-card">
            <span class="card-fa-icon"><i class="fa-solid {icon}"></i></span>
            <div class="card-label">{label}</div>
            <div class="card-value">{value}</div>
            <div class="card-delta {cls}">{sub}</div>
        </div>
        """, unsafe_allow_html=True)

st.divider()

# ── Satisfaction Distribution ─────────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="section-accent"></div>
    <h2>Satisfaction Score Distribution</h2>
</div>
""", unsafe_allow_html=True)

SAT_COLORS = {"Low Satisfaction": "#EF4444", "Moderate Satisfaction": "#F59E0B", "High Satisfaction": "#10B981"}
sat_map    = {0: "Low Satisfaction", 1: "Moderate Satisfaction", 2: "High Satisfaction"}
sat_counts = df["Satisfaction_Class"].map(sat_map).value_counts()

sd1, sd2 = st.columns(2, gap="small")
with sd1:
    fig_pie = px.pie(values=sat_counts.values, names=sat_counts.index,
                     color=sat_counts.index, color_discrete_map=SAT_COLORS, hole=0.45)
    fig_pie.update_traces(textposition="outside", textinfo="percent+label",
                          textfont=dict(size=11, color="#94A3B8"),
                          marker=dict(line=dict(color="#111827", width=2)))
    lay = get_chart_layout(height=380, title="Satisfaction Class Breakdown")
    lay["xaxis"] = dict(visible=False); lay["yaxis"] = dict(visible=False)
    fig_pie.update_layout(**lay)
    st.plotly_chart(fig_pie, use_container_width=True)

with sd2:
    fig_hist = px.histogram(df, x="Satisfaction_Score", nbins=25,
                            color_discrete_sequence=["#6366F1"],
                            labels={"Satisfaction_Score": "Score (1–5)", "count": "Customers"})
    fig_hist.update_traces(marker=dict(line=dict(color="#111827", width=1)))
    fig_hist.update_layout(**get_chart_layout(height=380, show_legend=False, title="Satisfaction Score Histogram"))
    st.plotly_chart(fig_hist, use_container_width=True)

st.divider()

# ── Category Analysis ─────────────────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="section-accent"></div>
    <h2>Purchase Category Analysis</h2>
</div>
""", unsafe_allow_html=True)

CAT_COLORS = ["#6366F1", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6", "#06B6D4"]
cat1, cat2 = st.columns(2, gap="small")
with cat1:
    cat_counts = df["Purchase_Category"].value_counts()
    fig_cat = px.bar(x=cat_counts.index, y=cat_counts.values,
                     color=cat_counts.index, color_discrete_sequence=CAT_COLORS,
                     text=cat_counts.values, labels={"x": "Category", "y": "Customers"})
    fig_cat.update_traces(textposition="outside", marker=dict(line=dict(width=0)), width=0.5)
    fig_cat.update_layout(**get_chart_layout(height=380, show_legend=False, title="Customers by Purchase Category"))
    st.plotly_chart(fig_cat, use_container_width=True)

with cat2:
    cat_spend = df.groupby("Purchase_Category")["Avg_Spend"].mean().sort_values(ascending=False)
    fig_cspend = px.bar(x=cat_spend.index, y=cat_spend.values,
                        color=cat_spend.index, color_discrete_sequence=CAT_COLORS,
                        text=[f"${v:.0f}" for v in cat_spend.values],
                        labels={"x": "Category", "y": "Avg Spend ($)"})
    fig_cspend.update_traces(textposition="outside", marker=dict(line=dict(width=0)), width=0.5)
    fig_cspend.update_layout(**get_chart_layout(height=380, show_legend=False, title="Avg Spend by Category"))
    st.plotly_chart(fig_cspend, use_container_width=True)

st.divider()

# ── Region & Income ───────────────────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="section-accent"></div>
    <h2>Region &amp; Income Analysis</h2>
</div>
""", unsafe_allow_html=True)

ri1, ri2 = st.columns(2, gap="small")
REGION_COLORS = {"Urban": "#6366F1", "Semi-Urban": "#10B981", "Rural": "#F59E0B"}
INCOME_COLORS = {"High": "#10B981", "Medium": "#6366F1", "Low": "#EF4444"}

with ri1:
    region_spend = df.groupby("Region")["Avg_Spend"].mean()
    fig_reg = px.bar(x=region_spend.index, y=region_spend.values,
                     color=region_spend.index, color_discrete_map=REGION_COLORS,
                     text=[f"${v:.0f}" for v in region_spend.values],
                     labels={"x": "Region", "y": "Avg Spend ($)"})
    fig_reg.update_traces(textposition="outside", marker=dict(line=dict(width=0)), width=0.4)
    fig_reg.update_layout(**get_chart_layout(height=360, show_legend=False, title="Avg Spend by Region"))
    st.plotly_chart(fig_reg, use_container_width=True)

with ri2:
    income_sat = df.groupby("Income_Level")["Satisfaction_Score"].mean().reindex(["Low","Medium","High"])
    fig_inc = px.bar(x=income_sat.index, y=income_sat.values,
                     color=income_sat.index, color_discrete_map=INCOME_COLORS,
                     text=[f"{v:.2f}" for v in income_sat.values],
                     labels={"x": "Income Level", "y": "Avg Satisfaction Score"})
    fig_inc.update_traces(textposition="outside", marker=dict(line=dict(width=0)), width=0.4)
    fig_inc.update_layout(**get_chart_layout(height=360, show_legend=False, title="Satisfaction by Income Level"))
    st.plotly_chart(fig_inc, use_container_width=True)

st.divider()

# ── Payment Mode & Demographics ───────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="section-accent"></div>
    <h2>Payment Mode &amp; Demographics</h2>
</div>
""", unsafe_allow_html=True)

pd1, pd2 = st.columns(2, gap="small")
with pd1:
    pay_counts = df["Payment_Mode"].value_counts()
    fig_pay = px.pie(values=pay_counts.values, names=pay_counts.index,
                     hole=0.4, color_discrete_sequence=["#6366F1","#10B981","#F59E0B","#EF4444","#8B5CF6"])
    fig_pay.update_traces(textposition="outside", textinfo="percent+label",
                          textfont=dict(size=11, color="#94A3B8"),
                          marker=dict(line=dict(color="#111827", width=2)))
    prl = get_chart_layout(height=380, title="Payment Mode Distribution")
    prl["xaxis"] = dict(visible=False); prl["yaxis"] = dict(visible=False)
    fig_pay.update_layout(**prl)
    st.plotly_chart(fig_pay, use_container_width=True)

with pd2:
    fig_age = px.histogram(df, x="Age", nbins=25, color="Gender",
                           color_discrete_map={"F": "#EF4444", "M": "#6366F1", "Other": "#F59E0B"},
                           labels={"Age": "Age (years)", "count": "Customers"}, barmode="overlay", opacity=0.75)
    fig_age.update_layout(**get_chart_layout(height=380, show_legend=True, title="Age Distribution by Gender"))
    st.plotly_chart(fig_age, use_container_width=True)

st.divider()

# ── Spend vs Satisfaction Scatter ─────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="section-accent"></div>
    <h2>Spend vs Satisfaction Correlation</h2>
</div>
""", unsafe_allow_html=True)

fig_scatter = px.scatter(df.sample(min(2000, len(df)), random_state=42),
                         x="Avg_Spend", y="Satisfaction_Score",
                         color="Income_Level", color_discrete_map=INCOME_COLORS,
                         opacity=0.55, size_max=8,
                         hover_data=["Age", "Region", "Purchase_Category"],
                         labels={"Avg_Spend": "Avg Spend ($)", "Satisfaction_Score": "Satisfaction Score"})
z = np.polyfit(df["Avg_Spend"], df["Satisfaction_Score"], 1)
p = np.poly1d(z)
x_line = np.linspace(df["Avg_Spend"].min(), df["Avg_Spend"].max(), 100)
fig_scatter.add_trace(go.Scatter(x=x_line, y=p(x_line), mode="lines", name="Trend",
                                 line=dict(color="#6366F1", width=2.5, dash="dash")))
lay2 = get_chart_layout(height=460, show_legend=True, title="Avg Spend vs Satisfaction Score (by Income Level)")
lay2["hovermode"] = "closest"
fig_scatter.update_layout(**lay2)
st.plotly_chart(fig_scatter, use_container_width=True)

st.markdown("""
<div style="text-align:center;padding:1rem 0;color:#2D3B55;font-size:0.75rem;">
    Dashboard • 10,000 Consumer Behaviour Records
</div>
""", unsafe_allow_html=True)
