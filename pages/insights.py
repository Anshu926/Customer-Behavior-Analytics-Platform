"""
Business Insights — Strategic Analytics
Analyzes: Gender, Age, Income_Level, Region, Avg_Spend, Membership, Satisfaction_Score.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils import load_data, get_business_insights, render_sidebar, get_chart_layout, FA_CDN

st.set_page_config(page_title="Business Insights", page_icon="💡", layout="wide", initial_sidebar_state="expanded")
render_sidebar("insights")

@st.cache_data
def get_data_and_insights():
    df = load_data()
    return df, get_business_insights(df)

df, insights = get_data_and_insights()
total = len(df)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown(f"""
{FA_CDN}
<div class="hero-banner" style="padding:1.6rem 2rem;">
    <div class="hero-tag"><i class="fa-solid fa-lightbulb"></i> Data-Driven Intelligence</div>
    <h1>Business Insights &amp; Recommendations</h1>
    <p class="hero-sub">Automated analytics surfacing growth opportunities, membership impact, and strategic recommendations across 10,000 customer records.</p>
</div>
""", unsafe_allow_html=True)

# ── Snapshot KPIs ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="section-accent"></div>
    <h2>Key Insights at a Glance</h2>
</div>""", unsafe_allow_html=True)

k1, k2, k3, k4 = st.columns(4, gap="small")
snap = [
    (k1, "fa-face-smile",    "High Satisfaction",  f"{insights['high_sat_count']:,}",  "positive", f"{insights['high_sat_count']/total*100:.1f}% of customers"),
    (k2, "fa-face-meh",      "Moderate",           f"{insights['mod_sat_count']:,}",   "neutral",  f"{insights['mod_sat_count']/total*100:.1f}% of customers"),
    (k3, "fa-face-frown",    "Low Satisfaction",   f"{insights['low_sat_count']:,}",   "neutral",  f"{insights['low_sat_count']/total*100:.1f}% of customers"),
    (k4, "fa-chart-scatter", "Spend–Sat. Corr.",   f"{insights['spend_sat_corr']:.3f}","info",     "Pearson correlation"),
]
for col, icon, label, value, cls, delta in snap:
    with col:
        st.markdown(f"""
        <div class="pro-card">
            <span class="card-fa-icon"><i class="fa-solid {icon}"></i></span>
            <div class="card-label">{label}</div>
            <div class="card-value" style="font-size:1.6rem;">{value}</div>
            <div class="card-delta {cls}">{delta}</div>
        </div>""", unsafe_allow_html=True)

st.divider()

# ── Insight 1: Membership Impact ──────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="section-accent"></div>
    <h2>Insight 1 &middot; Membership Program Value</h2>
</div>""", unsafe_allow_html=True)

i1l, i1r = st.columns([1, 2], gap="small")
mem_spend = df.groupby("Membership")["Avg_Spend"].mean()
mem_sat   = df.groupby("Membership")["Satisfaction_Score"].mean()

with i1l:
    st.markdown(f"""
    <div class="insight-panel">
        <div class="ip-num">01 — MEMBERSHIP IMPACT</div>
        <h4>💳 {insights['membership_pct']:.1f}% Enrolled in Membership</h4>
        <p style="color:#94A3B8;font-size:0.84rem;line-height:1.6;">
            Enrolled members spend an average of <strong style="color:#34D399;">${mem_spend.get('Yes',0):.2f}</strong>
            vs <strong style="color:#FCA5A5;">${mem_spend.get('No',0):.2f}</strong> for non-members.
            Average satisfaction score: <strong style="color:#A5B4FC;">{mem_sat.get('Yes',0):.2f}</strong> vs {mem_sat.get('No',0):.2f}.
        </p>
        <div class="badge badge-green" style="font-size:0.7rem;">High Retention Driver</div>
    </div>""", unsafe_allow_html=True)

with i1r:
    mem_df = pd.DataFrame({
        "Status": ["Non-Member", "Member"],
        "Avg Spend": [mem_spend.get("No", 0), mem_spend.get("Yes", 0)],
        "Avg Satisfaction": [mem_sat.get("No", 0), mem_sat.get("Yes", 0)]
    })
    fig1 = px.bar(mem_df, x="Status", y="Avg Spend",
                  color="Status", color_discrete_map={"Non-Member": "#EF4444", "Member": "#10B981"},
                  text=mem_df["Avg Spend"].apply(lambda v: f"${v:.2f}"))
    fig1.update_traces(textposition="outside", width=0.35)
    fig1.update_layout(**get_chart_layout(height=280, show_legend=False, title="Spend: Members vs Non-Members"))
    st.plotly_chart(fig1, use_container_width=True)

st.divider()

# ── Insight 2: Income Dynamics ────────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="section-accent"></div>
    <h2>Insight 2 &middot; Income Tier Dynamics</h2>
</div>""", unsafe_allow_html=True)

i2l, i2r = st.columns([1, 2], gap="small")
inc_spend = df.groupby("Income_Level")["Avg_Spend"].mean().reindex(["Low", "Medium", "High"])
inc_sat   = df.groupby("Income_Level")["Satisfaction_Score"].mean().reindex(["Low", "Medium", "High"])

with i2l:
    st.markdown(f"""
    <div class="insight-panel">
        <div class="ip-num">02 — INCOME ANALYSIS</div>
        <h4>💰 {insights['top_income']} Income Leads Spend</h4>
        <p style="color:#94A3B8;font-size:0.84rem;line-height:1.6;">
            <strong>{insights['top_income']} Income</strong> customers spend an average of
            <strong style="color:#A5B4FC;">${insights['top_income_spend']:.2f}</strong>.
            Tailor premium membership upgrades to this segment to maximize customer lifetime value.
        </p>
        <div class="badge badge-indigo" style="font-size:0.7rem;">Value Segment</div>
    </div>""", unsafe_allow_html=True)

with i2r:
    fig2 = px.bar(x=inc_spend.index, y=inc_spend.values,
                  color=inc_spend.index, color_discrete_map={"High":"#10B981","Medium":"#6366F1","Low":"#EF4444"},
                  text=[f"${v:.0f}" for v in inc_spend.values],
                  labels={"x": "Income Level", "y": "Avg Spend ($)"})
    fig2.update_traces(textposition="outside", width=0.4)
    fig2.update_layout(**get_chart_layout(height=280, show_legend=False, title="Avg Spend by Income Level"))
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ── Insight 3: Region Performance ────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="section-accent"></div>
    <h2>Insight 3 &middot; Regional Spending &amp; Volume</h2>
</div>""", unsafe_allow_html=True)

i3l, i3r = st.columns([1, 2], gap="small")
with i3l:
    st.markdown(f"""
    <div class="insight-panel">
        <div class="ip-num">03 — REGIONAL LEADER</div>
        <h4>📍 {insights['top_region']}</h4>
        <p style="color:#94A3B8;font-size:0.84rem;line-height:1.6;">
            The <strong style="color:#A5B4FC;">{insights['top_region']}</strong> region leads in monetary volume with
            an average spend of <strong style="color:#34D399;">${insights['top_region_spend']:.2f}</strong>.
            Focus geo-targeted promotions and logistics optimization in this market.
        </p>
        <div class="badge badge-amber" style="font-size:0.7rem;">Priority Region</div>
    </div>""", unsafe_allow_html=True)

with i3r:
    reg_df = df.groupby("Region").agg(Avg_Spend=("Avg_Spend","mean"), Customers=("Avg_Spend","count")).reset_index()
    REGION_COLORS = {"Urban": "#6366F1", "Semi-Urban": "#10B981", "Rural": "#F59E0B"}
    fig3 = px.scatter(reg_df, x="Region", y="Avg_Spend", size="Customers",
                      color="Region", color_discrete_map=REGION_COLORS,
                      text="Region", size_max=50,
                      labels={"Avg_Spend": "Avg Spend ($)"})
    fig3.update_traces(textposition="top center")
    fig3.update_layout(**get_chart_layout(height=280, show_legend=False, title="Avg Spend & Customer Volume by Region"))
    st.plotly_chart(fig3, use_container_width=True)

st.divider()

# ── Insight 4: Gender Distribution & Satisfaction ────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="section-accent"></div>
    <h2>Insight 4 &middot; Gender Demographics &amp; Satisfaction</h2>
</div>""", unsafe_allow_html=True)

i4l, i4r = st.columns([1, 2], gap="small")
gen_spend = df.groupby("Gender")["Avg_Spend"].mean()
gen_sat   = df.groupby("Gender")["Satisfaction_Score"].mean()
gen_map   = {"F": "Female", "M": "Male", "Other": "Other"}

with i4l:
    st.markdown(f"""
    <div class="insight-panel">
        <div class="ip-num">04 — GENDER PATTERNS</div>
        <h4>👥 Demographics Analysis</h4>
        <p style="color:#94A3B8;font-size:0.84rem;line-height:1.6;">
            Satisfaction scores remain remarkably balanced across genders (F: {gen_sat.get('F',0):.2f}, M: {gen_sat.get('M',0):.2f}, Other: {gen_sat.get('Other',0):.2f}),
            showing strong universal brand appeal with tailored engagement opportunities.
        </p>
        <div class="badge badge-indigo" style="font-size:0.7rem;">Balanced Demographics</div>
    </div>""", unsafe_allow_html=True)

with i4r:
    fig4 = px.bar(x=[gen_map.get(k, k) for k in gen_sat.index], y=gen_sat.values,
                  color=gen_sat.index, color_discrete_sequence=["#EF4444", "#6366F1", "#F59E0B"],
                  text=[f"{v:.2f} / 5.0" for v in gen_sat.values],
                  labels={"x": "Gender", "y": "Avg Satisfaction Score"})
    fig4.update_traces(textposition="outside", width=0.4)
    fig4.update_layout(**get_chart_layout(height=280, show_legend=False, title="Avg Satisfaction Score by Gender"))
    st.plotly_chart(fig4, use_container_width=True)

st.divider()

# ── Insight 5: Age Group Cohorts ──────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="section-accent"></div>
    <h2>Insight 5 &middot; Age Group Cohort Behavior</h2>
</div>""", unsafe_allow_html=True)

df["Age_Group"] = pd.cut(df["Age"], bins=[17,25,35,45,55,70],
                         labels=["18–25","26–35","36–45","46–55","56–69"])
age_spend = df.groupby("Age_Group", observed=True)["Avg_Spend"].mean()
age_sat   = df.groupby("Age_Group", observed=True)["Satisfaction_Score"].mean()

ag1, ag2 = st.columns(2, gap="small")
with ag1:
    fig5a = px.bar(x=age_spend.index.astype(str), y=age_spend.values,
                   color=age_spend.index.astype(str),
                   color_discrete_sequence=["#6366F1","#10B981","#F59E0B","#EF4444","#8B5CF6"],
                   text=[f"${v:.0f}" for v in age_spend.values],
                   labels={"x": "Age Group", "y": "Avg Spend ($)"})
    fig5a.update_traces(textposition="outside", width=0.45)
    fig5a.update_layout(**get_chart_layout(height=320, show_legend=False, title="Avg Spend by Age Cohort"))
    st.plotly_chart(fig5a, use_container_width=True)

with ag2:
    fig5b = px.bar(x=age_sat.index.astype(str), y=age_sat.values,
                   color=age_sat.index.astype(str),
                   color_discrete_sequence=["#6366F1","#10B981","#F59E0B","#EF4444","#8B5CF6"],
                   text=[f"{v:.2f}" for v in age_sat.values],
                   labels={"x": "Age Group", "y": "Avg Satisfaction"})
    fig5b.update_traces(textposition="outside", width=0.45)
    fig5b.update_layout(**get_chart_layout(height=320, show_legend=False, title="Avg Satisfaction by Age Cohort"))
    st.plotly_chart(fig5b, use_container_width=True)

st.divider()

# ── Strategic Recommendations ─────────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="section-accent"></div>
    <h2>Strategic Recommendations</h2>
</div>""", unsafe_allow_html=True)

recs = [
    ("fa-id-card",      "#10B981", "Accelerate Membership Conversions",
     f"With {insights['membership_pct']:.1f}% membership penetration, convert non-members through welcome perks, discount incentives, and exclusive advantages.",
     "Growth", "High Impact"),
    ("fa-wallet",       "#6366F1", "High-Income Segment Engagement",
     f"Target <strong>{insights['top_income']} Income</strong> customers with premium packages and concierge support to drive further average spend gains.",
     "Revenue", "High Impact"),
    ("fa-map-location", "#F59E0B", "Regional Expansion Strategy",
     f"Capitalize on high spend in <strong>{insights['top_region']}</strong> with focused digital marketing and priority service channels.",
     "Expansion", "Medium Impact"),
    ("fa-triangle-exclamation", "#EF4444", "Address Low Satisfaction Cohort",
     f"{insights['low_sat_pct']:.1f}% of customers fall into the low satisfaction tier. Launch proactive survey outreach and instant support resolution.",
     "Retention", "Critical"),
]
for icon, clr, title, desc, tag, priority in recs:
    st.markdown(f"""
    <div class="rec-card">
        <div class="rc-title"><i class="fa-solid {icon}" style="color:{clr};margin-right:0.5rem;"></i>{title}</div>
        <div class="rc-desc">{desc}</div>
        <div class="rc-meta">
            <span class="badge badge-indigo">{tag}</span>
            <span class="badge badge-{'red' if priority=='Critical' else 'amber' if priority=='Medium Impact' else 'green'}">{priority}</span>
        </div>
    </div>""", unsafe_allow_html=True)
