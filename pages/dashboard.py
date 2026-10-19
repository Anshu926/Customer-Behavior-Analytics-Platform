"""
Business Dashboard — KPIs and visualizations
Analyzes: Gender, Age, Income_Level, Region, Avg_Spend, Membership, Satisfaction_Score.
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
    <p class="hero-sub">Executive metrics and customer intelligence across Demographics, Region, Income Level, Spend, and Membership status.</p>
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
    (k1, "fa-users",          "Total Customers",   f"{stats['total_customers']:,}",      "positive", "Active records"),
    (k2, "fa-dollar-sign",    "Avg Spend",         f"${stats['avg_spend']:.2f}",          "positive", "Per customer"),
    (k3, "fa-star",           "Avg Satisfaction",  f"{stats['avg_score']:.2f} / 5.0",    "neutral",  "Target metric"),
    (k4, "fa-arrow-trend-up", "Total Revenue",     f"${stats['total_revenue']:,.0f}",     "positive", "Combined spend"),
    (k5, "fa-id-card",        "Membership Rate",   f"{stats['membership_pct']:.1f}%",    "positive", "Enrolled members"),
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
    lay = get_chart_layout(height=380, title="Satisfaction Tier Breakdown")
    lay["xaxis"] = dict(visible=False); lay["yaxis"] = dict(visible=False)
    fig_pie.update_layout(**lay)
    st.plotly_chart(fig_pie, use_container_width=True)

with sd2:
    fig_hist = px.histogram(df, x="Satisfaction_Score", nbins=25,
                            color_discrete_sequence=["#6366F1"],
                            labels={"Satisfaction_Score": "Satisfaction Score (1.0–5.0)", "count": "Customers"})
    fig_hist.update_traces(marker=dict(line=dict(color="#111827", width=1)))
    fig_hist.update_layout(**get_chart_layout(height=380, show_legend=False, title="Satisfaction Score Distribution"))
    st.plotly_chart(fig_hist, use_container_width=True)

st.divider()

# ── Membership Analysis ───────────────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="section-accent"></div>
    <h2>Membership Impact Analysis</h2>
</div>
""", unsafe_allow_html=True)

mem1, mem2 = st.columns(2, gap="small")
MEM_COLORS = {"Yes": "#10B981", "No": "#EF4444"}

with mem1:
    mem_spend = df.groupby("Membership")["Avg_Spend"].mean()
    fig_m_spend = px.bar(x=mem_spend.index, y=mem_spend.values,
                         color=mem_spend.index, color_discrete_map=MEM_COLORS,
                         text=[f"${v:.2f}" for v in mem_spend.values],
                         labels={"x": "Membership Status", "y": "Avg Spend ($)"})
    fig_m_spend.update_traces(textposition="outside", width=0.4)
    fig_m_spend.update_layout(**get_chart_layout(height=360, show_legend=False, title="Avg Spend: Members vs Non-Members"))
    st.plotly_chart(fig_m_spend, use_container_width=True)

with mem2:
    mem_sat = df.groupby("Membership")["Satisfaction_Score"].mean()
    fig_m_sat = px.bar(x=mem_sat.index, y=mem_sat.values,
                       color=mem_sat.index, color_discrete_map=MEM_COLORS,
                       text=[f"{v:.2f} / 5.0" for v in mem_sat.values],
                       labels={"x": "Membership Status", "y": "Avg Satisfaction Score"})
    fig_m_sat.update_traces(textposition="outside", width=0.4)
    fig_m_sat.update_layout(**get_chart_layout(height=360, show_legend=False, title="Avg Satisfaction: Members vs Non-Members"))
    st.plotly_chart(fig_m_sat, use_container_width=True)

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
    fig_reg.update_traces(textposition="outside", width=0.4)
    fig_reg.update_layout(**get_chart_layout(height=360, show_legend=False, title="Avg Spend by Region"))
    st.plotly_chart(fig_reg, use_container_width=True)

with ri2:
    income_sat = df.groupby("Income_Level")["Satisfaction_Score"].mean().reindex(["Low","Medium","High"])
    fig_inc = px.bar(x=income_sat.index, y=income_sat.values,
                     color=income_sat.index, color_discrete_map=INCOME_COLORS,
                     text=[f"{v:.2f}" for v in income_sat.values],
                     labels={"x": "Income Level", "y": "Avg Satisfaction Score"})
    fig_inc.update_traces(textposition="outside", width=0.4)
    fig_inc.update_layout(**get_chart_layout(height=360, show_legend=False, title="Satisfaction by Income Level"))
    st.plotly_chart(fig_inc, use_container_width=True)

st.divider()

# ── Gender & Age Demographics ─────────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="section-accent"></div>
    <h2>Demographics Overview</h2>
</div>
""", unsafe_allow_html=True)

pd1, pd2 = st.columns(2, gap="small")
GENDER_COLORS = {"F": "#EF4444", "M": "#6366F1", "Other": "#F59E0B"}

with pd1:
    gender_counts = df["Gender"].value_counts()
    gender_labels = {"F": "Female", "M": "Male", "Other": "Other"}
    fig_gen = px.pie(values=gender_counts.values,
                     names=[gender_labels.get(k, k) for k in gender_counts.index],
                     hole=0.4, color_discrete_sequence=["#6366F1","#EF4444","#F59E0B"])
    fig_gen.update_traces(textposition="outside", textinfo="percent+label",
                          textfont=dict(size=11, color="#94A3B8"),
                          marker=dict(line=dict(color="#111827", width=2)))
    g_lay = get_chart_layout(height=360, title="Gender Distribution")
    g_lay["xaxis"] = dict(visible=False); g_lay["yaxis"] = dict(visible=False)
    fig_gen.update_layout(**g_lay)
    st.plotly_chart(fig_gen, use_container_width=True)

with pd2:
    fig_age = px.histogram(df, x="Age", nbins=25, color="Gender",
                           color_discrete_map=GENDER_COLORS,
                           labels={"Age": "Age (years)", "count": "Customers"}, barmode="overlay", opacity=0.75)
    fig_age.update_layout(**get_chart_layout(height=360, show_legend=True, title="Age Distribution by Gender"))
    st.plotly_chart(fig_age, use_container_width=True)

st.divider()

# ── Spend vs Satisfaction Scatter ─────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="section-accent"></div>
    <h2>Spend vs Satisfaction Correlation</h2>
</div>
""", unsafe_allow_html=True)

fig_scatter = px.scatter(df.sample(min(2500, len(df)), random_state=42),
                         x="Avg_Spend", y="Satisfaction_Score",
                         color="Membership", color_discrete_map={"Yes": "#10B981", "No": "#EF4444"},
                         opacity=0.6,
                         hover_data=["Age", "Gender", "Income_Level", "Region"],
                         labels={"Avg_Spend": "Avg Spend ($)", "Satisfaction_Score": "Satisfaction Score"})
z = np.polyfit(df["Avg_Spend"], df["Satisfaction_Score"], 1)
p = np.poly1d(z)
x_line = np.linspace(df["Avg_Spend"].min(), df["Avg_Spend"].max(), 100)
fig_scatter.add_trace(go.Scatter(x=x_line, y=p(x_line), mode="lines", name="Trend",
                                 line=dict(color="#6366F1", width=2.5, dash="dash")))
lay2 = get_chart_layout(height=460, show_legend=True, title="Avg Spend vs Satisfaction Score (Colored by Membership)")
lay2["hovermode"] = "closest"
fig_scatter.update_layout(**lay2)
st.plotly_chart(fig_scatter, use_container_width=True)

st.markdown("""
<div style="text-align:center;padding:1rem 0;color:#2D3B55;font-size:0.75rem;">
    Dashboard • 10,000 Customer Records
</div>
""", unsafe_allow_html=True)
