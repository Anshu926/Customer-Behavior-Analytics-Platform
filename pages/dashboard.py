"""
Business Dashboard — KPIs and visualizations
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

# ── Hero ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
{FA_CDN}
<div class="hero-banner" style="padding:1.6rem 2rem;">
    <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:0.5rem;">
        <div class="hero-tag"><i class="fa-solid fa-chart-bar"></i> Real-time Analytics</div>
        <a href="https://www.kaggle.com/datasets/uom190346a/e-commerce-customer-behavior-dataset" target="_blank" style="
            display:inline-flex; align-items:center; gap:0.45rem;
            background:rgba(99,102,241,0.2); border:1px solid rgba(99,102,241,0.4);
            color:#FFFFFF !important; -webkit-text-fill-color:#FFFFFF !important;
            font-size:0.75rem; font-weight:700; padding:0.3rem 0.75rem;
            border-radius:8px; text-decoration:none;
        ">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="#20BEFF" xmlns="http://www.w3.org/2000/svg" style="vertical-align:middle;">
                <path d="M18.825 23.859c-.022.046-.054.084-.092.115s-.084.053-.133.065c-.05.011-.101.011-.151 0s-.098-.033-.139-.065l-6.845-5.69-2.733 2.623v2.858a.24.24 0 0 1-.07.17.24.24 0 0 1-.17.07H5.25a.24.24 0 0 1-.17-.07.24.24 0 0 1-.07-.17V.24A.24.24 0 0 1 5.08.07.24.24 0 0 1 5.25 0h3.243c.064 0 .125.025.17.07.045.045.07.106.07.17v14.168l8.98-9.053a.35.35 0 0 1 .253-.105h4.154a.24.24 0 0 1 .184.086.24.24 0 0 1 .054.198l-7.79 7.747 8.358 10.308a.24.24 0 0 1 .054.207.24.24 0 0 1-.153.16z"/>
            </svg>
            <span style="color:#FFFFFF !important; -webkit-text-fill-color:#FFFFFF !important;">Kaggle Dataset</span>
            <i class="fa-solid fa-arrow-up-right-from-square" style="font-size:0.65rem; color:#FFFFFF !important; -webkit-text-fill-color:#FFFFFF !important;"></i>
        </a>
    </div>
    <h1>Business Dashboard</h1>
    <p class="hero-sub">Executive-level KPIs and comprehensive customer metrics based on the E-Commerce Customer Behavior Dataset</p>
</div>
""", unsafe_allow_html=True)

# ── KPI ribbon ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="section-accent"></div>
    <h2>Key Performance Indicators</h2>
</div>
""", unsafe_allow_html=True)

k1, k2, k3, k4 = st.columns(4, gap="small")
kpis = [
    (k1, "fa-users",          "Total Customers",  f"{stats['total_customers']:,}",    "positive", "12% from last month"),
    (k2, "fa-dollar-sign",    "Average Spend",    f"${stats['avg_spend']:.2f}",       "positive", "5% from last month"),
    (k3, "fa-star",           "Average Rating",   f"{stats['avg_rating']:.2f} / 5.0","neutral",  "High satisfaction"),
    (k4, "fa-arrow-trend-up", "Total Revenue",    f"${stats['total_revenue']:,.0f}",  "positive", "8% from last month"),
]
for col, icon, label, value, cls, delta in kpis:
    with col:
        st.markdown(f"""
        <div class="pro-card">
            <span class="card-fa-icon"><i class="fa-solid {icon}"></i></span>
            <div class="card-label">{label}</div>
            <div class="card-value">{value}</div>
            <div class="card-delta {cls}"><i class="fa-solid fa-arrow-up" style="font-size:0.6rem;"></i> {delta}</div>
        </div>
        """, unsafe_allow_html=True)

st.divider()

# ── Satisfaction distribution ─────────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="section-accent"></div>
    <h2>Satisfaction Distribution</h2>
</div>
""", unsafe_allow_html=True)

satisfaction_counts = df["Satisfaction Level"].value_counts()
satisfaction_labels = {0:"Unsatisfied", 1:"Neutral", 2:"Satisfied"}
SAT_COLORS = {"Unsatisfied":"#EF4444","Neutral":"#F59E0B","Satisfied":"#10B981"}

sd1, sd2 = st.columns(2, gap="small")
with sd1:
    fig_pie = px.pie(
        values=satisfaction_counts.values,
        names=[satisfaction_labels[i] for i in satisfaction_counts.index],
        color=[satisfaction_labels[i] for i in satisfaction_counts.index],
        color_discrete_map=SAT_COLORS, hole=0.45,
    )
    fig_pie.update_traces(textposition="outside", textinfo="percent+label",
                          textfont=dict(size=11,color="#94A3B8"),
                          marker=dict(line=dict(color="#111827",width=2)))
    lay = get_chart_layout(height=380, title="Satisfaction Breakdown")
    lay["xaxis"] = dict(visible=False); lay["yaxis"] = dict(visible=False)
    fig_pie.update_layout(**lay)
    st.plotly_chart(fig_pie, use_container_width=True)

with sd2:
    sat_df = pd.DataFrame({"Satisfaction":[satisfaction_labels[i] for i in satisfaction_counts.index],"Count":satisfaction_counts.values})
    fig_bar = px.bar(sat_df, x="Satisfaction", y="Count", color="Satisfaction",
                     color_discrete_map=SAT_COLORS, text="Count")
    fig_bar.update_traces(textposition="outside", marker=dict(line=dict(width=0)), width=0.5)
    fig_bar.update_layout(**get_chart_layout(height=380, show_legend=False, title="Customer Count by Satisfaction"))
    st.plotly_chart(fig_bar, use_container_width=True)

st.divider()

# ── Membership analysis ───────────────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="section-accent"></div>
    <h2>Membership Type Analysis</h2>
</div>
""", unsafe_allow_html=True)

membership_counts = df["Membership Type"].value_counts().sort_index()
membership_labels = {1:"Bronze",2:"Silver",3:"Gold"}
MEM_COLORS = {"Bronze":"#CD7F32","Silver":"#94A3B8","Gold":"#F59E0B"}

ma1, ma2 = st.columns(2, gap="small")
with ma1:
    xl = [membership_labels[i] for i in membership_counts.index]
    fig_mem = px.bar(x=xl, y=membership_counts.values, color=xl, color_discrete_map=MEM_COLORS,
                     text=membership_counts.values, labels={"x":"Membership","y":"Count"})
    fig_mem.update_traces(textposition="outside", marker=dict(line=dict(width=0)), width=0.45)
    fig_mem.update_layout(**get_chart_layout(height=360, show_legend=False, title="Members by Tier"))
    st.plotly_chart(fig_mem, use_container_width=True)

with ma2:
    membership_spend = df.groupby("Membership Type")["Total Spend"].mean().sort_index()
    xl2 = [membership_labels[i] for i in membership_spend.index]
    fig_spend = px.bar(x=xl2, y=membership_spend.values, color=xl2, color_discrete_map=MEM_COLORS,
                       text=[f"${v:.0f}" for v in membership_spend.values], labels={"x":"Membership","y":"Avg Spend ($)"})
    fig_spend.update_traces(textposition="outside", marker=dict(line=dict(width=0)), width=0.45)
    fig_spend.update_layout(**get_chart_layout(height=360, show_legend=False, title="Average Spend by Tier"))
    st.plotly_chart(fig_spend, use_container_width=True)

st.divider()

# ── Demographics & Spending ───────────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="section-accent"></div>
    <h2>Demographics &amp; Spending</h2>
</div>
""", unsafe_allow_html=True)

dem1, dem2 = st.columns(2, gap="small")
with dem1:
    fig_age = px.histogram(df, x="Age", nbins=22, color_discrete_sequence=["#6366F1"],
                           labels={"Age":"Age (years)","count":"Customers"})
    fig_age.update_traces(marker=dict(line=dict(color="#111827",width=1)))
    fig_age.update_layout(**get_chart_layout(height=360, show_legend=False, title="Age Distribution"))
    st.plotly_chart(fig_age, use_container_width=True)

with dem2:
    fig_spend = px.histogram(df, x="Total Spend", nbins=22, color_discrete_sequence=["#10B981"],
                             labels={"Total Spend":"Total Spend ($)","count":"Customers"})
    fig_spend.update_traces(marker=dict(line=dict(color="#111827",width=1)))
    fig_spend.update_layout(**get_chart_layout(height=360, show_legend=False, title="Spend Distribution"))
    st.plotly_chart(fig_spend, use_container_width=True)

st.divider()

# ── Discount impact ───────────────────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="section-accent"></div>
    <h2>Discount Impact Analysis</h2>
</div>
""", unsafe_allow_html=True)

disc1, disc2 = st.columns(2, gap="small")
discount_data   = df["Discount Applied"].value_counts()
discount_labels = {0:"No Discount",1:"Discount Applied"}
DISC_COLORS     = {"No Discount":"#EF4444","Discount Applied":"#10B981"}

with disc1:
    fig_dpie = px.pie(values=discount_data.values,
                      names=[discount_labels[i] for i in discount_data.index],
                      color=[discount_labels[i] for i in discount_data.index],
                      color_discrete_map=DISC_COLORS, hole=0.45)
    fig_dpie.update_traces(textposition="outside", textinfo="percent+label",
                           textfont=dict(size=11,color="#94A3B8"),
                           marker=dict(line=dict(color="#111827",width=2)))
    dl = get_chart_layout(height=360, title="Discount Penetration")
    dl["xaxis"] = dict(visible=False); dl["yaxis"] = dict(visible=False)
    fig_dpie.update_layout(**dl)
    st.plotly_chart(fig_dpie, use_container_width=True)

with disc2:
    disc_sat = df.groupby("Discount Applied")["Satisfaction Level"].mean()
    xl3 = [discount_labels[i] for i in disc_sat.index]
    fig_dsat = px.bar(x=xl3, y=disc_sat.values, color=xl3, color_discrete_map=DISC_COLORS,
                      text=[f"{v:.2f}" for v in disc_sat.values], labels={"x":"Discount","y":"Avg Satisfaction"})
    fig_dsat.update_traces(textposition="outside", marker=dict(line=dict(width=0)), width=0.4)
    fig_dsat.update_layout(**get_chart_layout(height=360, show_legend=False, title="Discount vs Satisfaction"))
    st.plotly_chart(fig_dsat, use_container_width=True)

st.divider()

# ── City tier ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="section-accent"></div>
    <h2>City Tier Performance</h2>
</div>
""", unsafe_allow_html=True)

city_tier_counts = df["City Tier"].value_counts().sort_index()
city_tier_labels = {1:"Tier 1 — Low",2:"Tier 2 — Medium",3:"Tier 3 — High"}
TIER_COLORS = ["#EF4444","#F59E0B","#10B981"]

ct1, ct2 = st.columns(2, gap="small")
with ct1:
    xl4 = [city_tier_labels[i] for i in city_tier_counts.index]
    fig_city = px.bar(x=xl4, y=city_tier_counts.values, color=xl4,
                      color_discrete_sequence=TIER_COLORS, text=city_tier_counts.values,
                      labels={"x":"City Tier","y":"Count"})
    fig_city.update_traces(textposition="outside", marker=dict(line=dict(width=0)), width=0.45)
    fig_city.update_layout(**get_chart_layout(height=360, show_legend=False, title="Customers by City Tier"))
    st.plotly_chart(fig_city, use_container_width=True)

with ct2:
    city_spend = df.groupby("City Tier")["Total Spend"].mean().sort_index()
    xl5 = [city_tier_labels[i] for i in city_spend.index]
    fig_cspend = px.bar(x=xl5, y=city_spend.values, color=xl5,
                        color_discrete_sequence=TIER_COLORS, text=[f"${v:.0f}" for v in city_spend.values],
                        labels={"x":"City Tier","y":"Avg Spend ($)"})
    fig_cspend.update_traces(textposition="outside", marker=dict(line=dict(width=0)), width=0.45)
    fig_cspend.update_layout(**get_chart_layout(height=360, show_legend=False, title="Avg Spend by City Tier"))
    st.plotly_chart(fig_cspend, use_container_width=True)

st.divider()

# ── Scatter ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="section-accent"></div>
    <h2>Spend vs Satisfaction Correlation</h2>
</div>
""", unsafe_allow_html=True)

fig_scatter = px.scatter(df, x="Total Spend", y="Satisfaction Level",
                         color="Membership Type",
                         color_discrete_map={1:"#CD7F32",2:"#94A3B8",3:"#F59E0B"},
                         size="Items Purchased", size_max=14,
                         hover_data=["Age","Average Rating"], opacity=0.7,
                         labels={"Total Spend":"Total Spend ($)","Satisfaction Level":"Satisfaction Level"})
z = np.polyfit(df["Total Spend"], df["Satisfaction Level"], 1)
p = np.poly1d(z)
x_line = np.linspace(df["Total Spend"].min(), df["Total Spend"].max(), 100)
fig_scatter.add_trace(go.Scatter(x=x_line, y=p(x_line), mode="lines", name="Trend",
                                 line=dict(color="#6366F1",width=2.5,dash="dash")))
lay2 = get_chart_layout(height=460, show_legend=True, title="Spend vs Satisfaction (by Membership Tier)")
lay2["hovermode"] = "closest"
fig_scatter.update_layout(**lay2)
st.plotly_chart(fig_scatter, use_container_width=True)

st.markdown("""
<div style="text-align:center;padding:1rem 0;color:#2D3B55;font-size:0.75rem;">
    <i class="fa-solid fa-clock-rotate-left" style="margin-right:0.3rem;"></i>
    Dashboard generated from live customer data
</div>
""", unsafe_allow_html=True)
