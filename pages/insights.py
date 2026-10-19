"""
Business Insights — 10k Consumer Behaviour Dataset
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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
    <p class="hero-sub">Automated analytics surfacing growth opportunities, retention risks, and strategic recommendations — powered by 10,000 consumer behaviour records.</p>
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
    (k4, "fa-chart-scatter", "Spend–Sat. Corr.",   f"{insights['spend_sat_corr']:.3f}","info",     "Pearson coefficient"),
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

# ── Insight 1: Top Category ───────────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="section-accent"></div>
    <h2>Insight 1 &middot; Category Revenue Leaders</h2>
</div>""", unsafe_allow_html=True)

i1l, i1r = st.columns([1, 2], gap="small")
with i1l:
    st.markdown(f"""
    <div class="insight-panel">
        <div class="ip-num">01 — TOP CATEGORY</div>
        <h4>🏆 {insights['top_category']}</h4>
        <p style="color:#94A3B8;font-size:0.84rem;line-height:1.6;">
            This is the highest-spending category with an average spend of
            <strong style="color:#A5B4FC;">${insights['top_category_spend']:.2f}</strong> per customer.
            Prioritise inventory and marketing efforts here.
        </p>
        <div class="badge badge-indigo" style="font-size:0.7rem;">Revenue Leader</div>
    </div>""", unsafe_allow_html=True)

with i1r:
    cat_spend = df.groupby("Purchase_Category")["Avg_Spend"].mean().sort_values(ascending=False)
    fig = px.bar(x=cat_spend.index, y=cat_spend.values, color=cat_spend.index,
                 color_discrete_sequence=["#6366F1","#10B981","#F59E0B","#EF4444","#8B5CF6","#06B6D4"],
                 text=[f"${v:.0f}" for v in cat_spend.values],
                 labels={"x": "Category", "y": "Avg Spend ($)"})
    fig.update_traces(textposition="outside", marker=dict(line=dict(width=0)), width=0.5)
    fig.update_layout(**get_chart_layout(height=300, show_legend=False))
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Insight 2: Loyalty Impact ─────────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="section-accent"></div>
    <h2>Insight 2 &middot; Loyalty Member Impact</h2>
</div>""", unsafe_allow_html=True)

i2l, i2r = st.columns([1, 2], gap="small")
loyalty_spend = df.groupby("Loyalty_Member")["Avg_Spend"].mean()
loyalty_sat   = df.groupby("Loyalty_Member")["Satisfaction_Score"].mean()

with i2l:
    loy_pct = insights["loyalty_pct"]
    st.markdown(f"""
    <div class="insight-panel">
        <div class="ip-num">02 — LOYALTY PROGRAMME</div>
        <h4>❤️ {loy_pct:.1f}% Are Loyal Members</h4>
        <p style="color:#94A3B8;font-size:0.84rem;line-height:1.6;">
            Loyal members spend <strong style="color:#34D399;">${loyalty_spend.get('Yes',0):.0f}</strong> on average,
            vs <strong style="color:#FCA5A5;">${loyalty_spend.get('No',0):.0f}</strong> for non-members.
            Satisfaction score: <strong style="color:#A5B4FC;">{loyalty_sat.get('Yes',0):.2f}</strong> vs {loyalty_sat.get('No',0):.2f}.
        </p>
        <div class="badge badge-green" style="font-size:0.7rem;">High Retention Value</div>
    </div>""", unsafe_allow_html=True)

with i2r:
    loy_df = pd.DataFrame({
        "Loyalty": ["Non-Member", "Loyal Member"],
        "Avg Spend": [loyalty_spend.get("No", 0), loyalty_spend.get("Yes", 0)],
        "Avg Satisfaction": [loyalty_sat.get("No", 0), loyalty_sat.get("Yes", 0)],
    })
    fig2 = px.bar(loy_df, x="Loyalty", y="Avg Spend",
                  color="Loyalty", color_discrete_map={"Non-Member":"#EF4444","Loyal Member":"#10B981"},
                  text=loy_df["Avg Spend"].apply(lambda v: f"${v:.0f}"))
    fig2.update_traces(textposition="outside", marker=dict(line=dict(width=0)), width=0.35)
    fig2.update_layout(**get_chart_layout(height=300, show_legend=False, title="Avg Spend: Loyal vs Non-Loyal"))
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ── Insight 3: Region Performance ────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="section-accent"></div>
    <h2>Insight 3 &middot; Regional Spending Performance</h2>
</div>""", unsafe_allow_html=True)

i3l, i3r = st.columns([1, 2], gap="small")
with i3l:
    st.markdown(f"""
    <div class="insight-panel">
        <div class="ip-num">03 — TOP REGION</div>
        <h4>📍 {insights['top_region']}</h4>
        <p style="color:#94A3B8;font-size:0.84rem;line-height:1.6;">
            The <strong style="color:#A5B4FC;">{insights['top_region']}</strong> region leads with an average spend of
            <strong style="color:#34D399;">${insights['top_region_spend']:.2f}</strong>.
            Focus digital campaigns and premium delivery here.
        </p>
        <div class="badge badge-amber" style="font-size:0.7rem;">High-Value Region</div>
    </div>""", unsafe_allow_html=True)

with i3r:
    reg_df = df.groupby("Region").agg(Avg_Spend=("Avg_Spend","mean"), Customers=("Avg_Spend","count")).reset_index()
    REGION_COLORS = {"Urban": "#6366F1", "Semi-Urban": "#10B981", "Rural": "#F59E0B"}
    fig3 = px.scatter(reg_df, x="Region", y="Avg_Spend", size="Customers",
                      color="Region", color_discrete_map=REGION_COLORS,
                      text="Region", size_max=60,
                      labels={"Avg_Spend": "Avg Spend ($)"})
    fig3.update_traces(textposition="top center")
    fig3.update_layout(**get_chart_layout(height=300, show_legend=False, title="Avg Spend & Volume by Region"))
    st.plotly_chart(fig3, use_container_width=True)

st.divider()

# ── Insight 4: Payment Mode ───────────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="section-accent"></div>
    <h2>Insight 4 &middot; Payment Mode &amp; Satisfaction</h2>
</div>""", unsafe_allow_html=True)

pay_sat = df.groupby("Payment_Mode")["Satisfaction_Score"].mean().sort_values(ascending=False)
pay_cnt = df["Payment_Mode"].value_counts()
fig4 = px.bar(x=pay_sat.index, y=pay_sat.values,
              color=pay_sat.index,
              color_discrete_sequence=["#6366F1","#10B981","#F59E0B","#EF4444","#8B5CF6"],
              text=[f"{v:.2f}" for v in pay_sat.values],
              labels={"x": "Payment Mode", "y": "Avg Satisfaction Score"})
fig4.update_traces(textposition="outside", marker=dict(line=dict(width=0)), width=0.5)
fig4.update_layout(**get_chart_layout(height=340, show_legend=False, title="Avg Satisfaction Score by Payment Mode"))
st.plotly_chart(fig4, use_container_width=True)

st.divider()

# ── Insight 5: Age-group analysis ─────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="section-accent"></div>
    <h2>Insight 5 &middot; Age Group Spending Patterns</h2>
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
    fig5a.update_traces(textposition="outside", marker=dict(line=dict(width=0)), width=0.5)
    fig5a.update_layout(**get_chart_layout(height=340, show_legend=False, title="Avg Spend by Age Group"))
    st.plotly_chart(fig5a, use_container_width=True)

with ag2:
    fig5b = px.bar(x=age_sat.index.astype(str), y=age_sat.values,
                   color=age_sat.index.astype(str),
                   color_discrete_sequence=["#6366F1","#10B981","#F59E0B","#EF4444","#8B5CF6"],
                   text=[f"{v:.2f}" for v in age_sat.values],
                   labels={"x": "Age Group", "y": "Avg Satisfaction"})
    fig5b.update_traces(textposition="outside", marker=dict(line=dict(width=0)), width=0.5)
    fig5b.update_layout(**get_chart_layout(height=340, show_legend=False, title="Avg Satisfaction by Age Group"))
    st.plotly_chart(fig5b, use_container_width=True)

st.divider()

# ── Strategic Recommendations ─────────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="section-accent"></div>
    <h2>Strategic Recommendations</h2>
</div>""", unsafe_allow_html=True)

recs = [
    ("fa-trophy",       "#10B981", "Invest in Top Categories",
     f"Prioritise <strong>{insights['top_category']}</strong> with targeted promotions and premium inventory — it drives the highest average spend.",
     "Revenue", "High Impact"),
    ("fa-heart",        "#6366F1", "Expand Loyalty Programme",
     f"With {insights['loyalty_pct']:.1f}% loyalty penetration, there is room to grow. Offer exclusive tiers, early access, and personalised deals to boost retention.",
     "Retention", "High Impact"),
    ("fa-map-location", "#F59E0B", "Regional Marketing Strategy",
     f"Double down on <strong>{insights['top_region']}</strong> campaigns. Consider premium delivery options and geo-targeted promotions for high-spend zones.",
     "Growth", "Medium Impact"),
    ("fa-triangle-exclamation", "#EF4444", "Address Low Satisfaction",
     f"{insights['low_sat_pct']:.1f}% of customers have low satisfaction. Activate targeted re-engagement campaigns with personalised offers and improved support.",
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
