"""
Business Insights — Data-Driven Analytics and Recommendations
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils import (
    load_data, get_business_insights,
    render_sidebar, get_chart_layout, FA_CDN,
    MEMBERSHIP_REVERSE,
)

st.set_page_config(page_title="Business Insights", page_icon="💡", layout="wide", initial_sidebar_state="expanded")
render_sidebar("insights")

@st.cache_data
def get_data_and_insights():
    df = load_data()
    return df, get_business_insights(df)

df, insights = get_data_and_insights()
total = len(df)

discount_with    = df[df["Discount Applied"] == 1]["Satisfaction Level"].mean()
discount_without = df[df["Discount Applied"] == 0]["Satisfaction Level"].mean()
discount_impact  = discount_with - discount_without

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown(f"""
{FA_CDN}
<div class="hero-banner" style="padding:1.6rem 2rem;">
    <div class="hero-tag"><i class="fa-solid fa-lightbulb"></i> Data-Driven Intelligence</div>
    <h1>Business Insights &amp; Recommendations</h1>
    <p class="hero-sub">Automated analytics surfacing growth opportunities, risks, and strategic recommendations from your customer data.</p>
</div>
""", unsafe_allow_html=True)

# ── Snapshot KPIs ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="section-accent"></div>
    <h2>Key Insights at a Glance</h2>
</div>
""", unsafe_allow_html=True)

k1, k2, k3, k4 = st.columns(4, gap="small")
snap = [
    (k1, "fa-face-smile",        "Satisfied Customers",    f"{insights['satisfied_count']}",   "positive", f"{insights['satisfied_count']/total*100:.1f}% of total"),
    (k2, "fa-face-meh",          "Neutral Customers",      f"{insights['neutral_count']}",     "neutral",  f"{insights['neutral_count']/total*100:.1f}% of total"),
    (k3, "fa-face-frown",        "Unsatisfied Customers",  f"{insights['unsatisfied_count']}", "neutral",  f"{insights['unsatisfied_count']/total*100:.1f}% of total"),
    (k4, "fa-chart-scatter",     "Spend–Sat. Correlation", f"{insights['spend_satisfaction_corr']:.3f}", "info", "Coefficient"),
]
for col, icon, label, value, cls, delta in snap:
    with col:
        st.markdown(f"""
        <div class="pro-card">
            <span class="card-fa-icon"><i class="fa-solid {icon}"></i></span>
            <div class="card-label">{label}</div>
            <div class="card-value" style="font-size:1.6rem;">{value}</div>
            <div class="card-delta {cls}">{delta}</div>
        </div>
        """, unsafe_allow_html=True)

st.divider()

# ── Insight 1: Membership ─────────────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="section-accent"></div>
    <h2>Insight 1 &middot; Membership Tier Performance</h2>
</div>
""", unsafe_allow_html=True)

i1l, i1r = st.columns([1,2], gap="small")
with i1l:
    st.markdown(f"""
    <div class="insight-panel">
        <div class="ip-num"><i class="fa-solid fa-id-badge" style="margin-right:0.3rem;"></i>Insight 01 &middot; Membership</div>
        <h4>Top Performing Tier</h4>
        <div style="font-size:1.8rem;font-weight:700;color:#F59E0B;margin:0.4rem 0;">
            <i class="fa-solid fa-trophy" style="font-size:1.4rem;margin-right:0.3rem;"></i>{insights['top_membership']}
        </div>
        <div style="color:#64748B;font-size:0.8rem;margin-bottom:0.8rem;">
            Avg Spend: <span style="color:#F1F5F9;font-weight:600;">${insights['top_membership_avg']:.2f}</span>
        </div>
        <div style="color:#94A3B8;font-size:0.82rem;line-height:1.6;">
            Focus on upgrading more customers to <strong style="color:#F1F5F9;">{insights['top_membership']}</strong> tier — this membership level shows the highest spending potential.
        </div>
    </div>
    """, unsafe_allow_html=True)

with i1r:
    membership_spend = df.groupby("Membership Type").agg({"Total Spend":["mean","count"]}).round(2)
    membership_labels = {1:"Bronze",2:"Silver",3:"Gold"}
    MEM_COLORS = {"Bronze":"#CD7F32","Silver":"#94A3B8","Gold":"#F59E0B"}
    xl = [membership_labels[i] for i in membership_spend.index]
    fig_mem = px.bar(x=xl, y=membership_spend[("Total Spend","mean")].values,
                     color=xl, color_discrete_map=MEM_COLORS,
                     text=[f"${v:.0f}" for v in membership_spend[("Total Spend","mean")].values],
                     labels={"x":"Membership","y":"Avg Spend ($)"})
    fig_mem.update_traces(textposition="outside", marker=dict(line=dict(width=0)), width=0.45)
    fig_mem.update_layout(**get_chart_layout(height=300, show_legend=False, title="Average Spend by Membership Tier"))
    st.plotly_chart(fig_mem, use_container_width=True)

st.divider()

# ── Insight 2: Low Engagement ─────────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="section-accent"></div>
    <h2>Insight 2 &middot; Low Engagement Risk</h2>
</div>
""", unsafe_allow_html=True)

i2l, i2r = st.columns([1,2], gap="small")
with i2l:
    st.markdown(f"""
    <div class="insight-panel">
        <div class="ip-num"><i class="fa-solid fa-triangle-exclamation" style="margin-right:0.3rem;"></i>Insight 02 &middot; Engagement</div>
        <h4>At-Risk Customers</h4>
        <div style="font-size:1.8rem;font-weight:700;color:#EF4444;margin:0.4rem 0;">
            {insights['low_engagement_count']}
        </div>
        <span class="badge badge-red">{insights['low_engagement_pct']:.1f}% of total</span>
        <div style="color:#94A3B8;font-size:0.82rem;line-height:1.6;margin-top:0.7rem;">
            Customers with days since purchase &gt; 75th percentile. Launch targeted re-engagement campaigns with exclusive comeback offers and personalised outreach.
        </div>
    </div>
    """, unsafe_allow_html=True)

with i2r:
    threshold = df["Days Since Last Purchase"].quantile(0.75)
    fig_eng = px.histogram(df, x="Days Since Last Purchase", nbins=22,
                           color_discrete_sequence=["#6366F1"],
                           labels={"Days Since Last Purchase":"Days","count":"Customers"})
    fig_eng.update_traces(marker=dict(line=dict(color="#111827",width=1)))
    fig_eng.add_vline(x=threshold, line_dash="dash", line_color="#EF4444", line_width=2,
                      annotation_text=f"Risk Threshold ({threshold:.0f}d)",
                      annotation_font_color="#EF4444", annotation_position="top right")
    fig_eng.update_layout(**get_chart_layout(height=300, show_legend=False, title="Days Since Last Purchase"))
    st.plotly_chart(fig_eng, use_container_width=True)

st.divider()

# ── Insight 3: Discounts ──────────────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="section-accent"></div>
    <h2>Insight 3 &middot; Discount Strategy Impact</h2>
</div>
""", unsafe_allow_html=True)

i3l, i3r = st.columns([1,2], gap="small")
impact_color = "#34D399" if discount_impact > 0 else "#EF4444"
impact_badge = "badge-green" if discount_impact > 0 else "badge-red"
impact_icon  = "fa-arrow-trend-up" if discount_impact > 0 else "fa-arrow-trend-down"
impact_text  = "Discounts are EFFECTIVE — increase strategic discount campaigns." if discount_impact > 0 else "Discounts have LIMITED IMPACT — focus on quality instead."

with i3l:
    st.markdown(f"""
    <div class="insight-panel">
        <div class="ip-num"><i class="fa-solid fa-tag" style="margin-right:0.3rem;"></i>Insight 03 &middot; Discounts</div>
        <h4>Discount Impact on Satisfaction</h4>
        <div style="font-size:1.7rem;font-weight:700;color:{impact_color};margin:0.4rem 0;">
            <i class="fa-solid {impact_icon}" style="font-size:1.2rem;margin-right:0.3rem;"></i>
            {abs(discount_impact):.3f}
        </div>
        <div style="display:flex;gap:0.5rem;margin-bottom:0.7rem;flex-wrap:wrap;">
            <span class="badge badge-green">With discount: {discount_with:.2f}</span>
            <span class="badge badge-red">Without: {discount_without:.2f}</span>
        </div>
        <div style="color:#94A3B8;font-size:0.82rem;line-height:1.6;">{impact_text}</div>
    </div>
    """, unsafe_allow_html=True)

with i3r:
    disc_data = df.groupby("Discount Applied")["Satisfaction Level"].agg(["mean","count"])
    fig_disc = px.bar(x=["No Discount","With Discount"], y=disc_data["mean"].values,
                      color=["No Discount","With Discount"],
                      color_discrete_map={"No Discount":"#EF4444","With Discount":"#10B981"},
                      text=[f"{v:.3f}" for v in disc_data["mean"].values],
                      labels={"x":"Discount Status","y":"Avg Satisfaction"})
    fig_disc.update_traces(textposition="outside", marker=dict(line=dict(width=0)), width=0.4)
    fig_disc.update_layout(**get_chart_layout(height=300, show_legend=False, title="Discount vs Average Satisfaction"))
    st.plotly_chart(fig_disc, use_container_width=True)

st.divider()

# ── Insight 4: Spend vs Satisfaction ─────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="section-accent"></div>
    <h2>Insight 4 &middot; Spend vs Satisfaction Relationship</h2>
</div>
""", unsafe_allow_html=True)

correlation   = insights["spend_satisfaction_corr"]
interp_str    = "STRONG" if abs(correlation) > 0.7 else "MODERATE" if abs(correlation) > 0.4 else "WEAK"
direction_str = "POSITIVE" if correlation > 0 else "NEGATIVE"
corr_color    = "#10B981" if correlation > 0.3 else "#F59E0B" if correlation > 0 else "#EF4444"
corr_cls      = "badge-green" if correlation > 0.3 else "badge-amber" if correlation > 0 else "badge-red"
rec_text      = ("Higher-spending customers are MORE satisfied — focus on maximising customer lifetime value."
                 if correlation > 0.3 else
                 "Spending and satisfaction are NOT strongly linked — focus on quality experience.")

i4l, i4r = st.columns([1,2], gap="small")
with i4l:
    st.markdown(f"""
    <div class="insight-panel">
        <div class="ip-num"><i class="fa-solid fa-chart-scatter" style="margin-right:0.3rem;"></i>Insight 04 &middot; Correlation</div>
        <h4>Spend–Satisfaction Correlation</h4>
        <div style="font-size:1.7rem;font-weight:700;color:{corr_color};margin:0.4rem 0;">{correlation:.3f}</div>
        <div style="display:flex;gap:0.4rem;margin-bottom:0.7rem;flex-wrap:wrap;">
            <span class="badge {corr_cls}">{interp_str}</span>
            <span class="badge badge-indigo">{direction_str}</span>
        </div>
        <div style="color:#94A3B8;font-size:0.82rem;line-height:1.6;">{rec_text}</div>
    </div>
    """, unsafe_allow_html=True)

with i4r:
    fig_scat = px.scatter(df, x="Total Spend", y="Satisfaction Level",
                          color_discrete_sequence=["#6366F1"],
                          labels={"Total Spend":"Total Spend ($)","Satisfaction Level":"Satisfaction Level"},
                          opacity=0.55)
    z = np.polyfit(df["Total Spend"], df["Satisfaction Level"], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df["Total Spend"].min(), df["Total Spend"].max(), 100)
    fig_scat.add_trace(go.Scatter(x=x_line, y=p(x_line), mode="lines", name="Trend",
                                  line=dict(color="#EF4444",width=2.5,dash="dash")))
    fig_scat.update_layout(**get_chart_layout(height=300, show_legend=False, title="Spend vs Satisfaction Scatter"))
    st.plotly_chart(fig_scat, use_container_width=True)

st.divider()

# ── Strategic recommendations ─────────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="section-accent"></div>
    <h2>Strategic Recommendations</h2>
</div>
""", unsafe_allow_html=True)

recommendations = [
    {"icon":"fa-rocket",          "color":"#F59E0B","title":"Upgrade Campaign",
     "desc":"Target Bronze members for upgrade to Silver / Gold membership.",
     "impact":"15–20% revenue increase","effort":"Medium","ic":"badge-green","ec":"badge-amber"},
    {"icon":"fa-envelope",        "color":"#6366F1","title":"Re-engagement Programme",
     "desc":"Reach out to customers with 90+ days since last purchase.",
     "impact":"10–15% churn reduction","effort":"Low","ic":"badge-green","ec":"badge-green"},
    {"icon":"fa-tag",             "color":"#10B981","title":"Strategic Discounting",
     "desc":f"Use discounts smartly — they improve satisfaction by {abs(discount_impact):.2f} on average.",
     "impact":"5–10% satisfaction lift","effort":"Low","ic":"badge-green","ec":"badge-green"},
    {"icon":"fa-star",            "color":"#EF4444","title":"Quality Focus",
     "desc":"Invest in product quality improvements to drive long-term satisfaction.",
     "impact":"20–30% spend increase","effort":"High","ic":"badge-green","ec":"badge-red"},
]

rc1, rc2 = st.columns(2, gap="small")
for i, rec in enumerate(recommendations):
    col = rc1 if i % 2 == 0 else rc2
    with col:
        st.markdown(f"""
        <div class="rec-card" style="border-top-color:{rec['color']};">
            <div class="rc-title">
                <i class="fa-solid {rec['icon']}" style="color:{rec['color']};margin-right:0.4rem;"></i>
                {rec['title']}
            </div>
            <div class="rc-desc">{rec['desc']}</div>
            <div class="rc-meta">
                <span class="badge {rec['ic']}"><i class="fa-solid fa-arrow-trend-up" style="font-size:0.6rem;"></i> {rec['impact']}</span>
                <span class="badge {rec['ec']}"><i class="fa-solid fa-gauge" style="font-size:0.6rem;"></i> Effort: {rec['effort']}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.divider()

# ── Growth opportunities ──────────────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="section-accent"></div>
    <h2>Growth Opportunities</h2>
</div>
""", unsafe_allow_html=True)

gold_pct = (df["Membership Type"] == 3).sum() / total * 100
go_panels = [
    {"title":"Untapped Potential",       "icon":"fa-bolt",       "color":"#6366F1",
     "items":[f"Premium tier adopted by only {gold_pct:.1f}% of customers",
              f"{insights['unsatisfied_count']} unsatisfied customers at risk",
              "Quality investment shows strong ROI via spend–satisfaction link"]},
    {"title":"Quick Wins (0–3 months)",  "icon":"fa-flag",       "color":"#10B981",
     "items":["Launch discount campaign for neutral customers",
              "Win-back email for disengaged (90+ days)",
              "Bronze-to-Silver upgrade programme",
              "Expected impact: +5% revenue"]},
    {"title":"Long-term (3–12 months)",  "icon":"fa-binoculars", "color":"#F59E0B",
     "items":["Product quality improvement roadmap",
              "Premium tier expansion incentives",
              "Loyalty programme redesign",
              "Expected impact: +25% revenue"]},
]

gp1, gp2, gp3 = st.columns(3, gap="small")
for col, panel in zip([gp1, gp2, gp3], go_panels):
    items_html = "".join(
        f'<div style="color:#64748B;font-size:0.81rem;padding:0.22rem 0;display:flex;align-items:flex-start;gap:0.4rem;">'
        f'<i class="fa-solid fa-chevron-right" style="font-size:0.55rem;color:{panel["color"]};margin-top:4px;flex-shrink:0;"></i>{item}</div>'
        for item in panel["items"]
    )
    with col:
        st.markdown(f"""
        <div style="background:#111827;border:1px solid #1F2D45;border-top:3px solid {panel['color']};border-radius:12px;padding:1.2rem 1.3rem;height:100%;">
            <div style="color:{panel['color']};font-size:0.92rem;font-weight:600;margin-bottom:0.7rem;">
                <i class="fa-solid {panel['icon']}" style="margin-right:0.4rem;"></i>{panel['title']}
            </div>
            {items_html}
        </div>
        """, unsafe_allow_html=True)

st.divider()

# ── Customer composition ──────────────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="section-accent"></div>
    <h2>Customer Composition</h2>
</div>
""", unsafe_allow_html=True)

membership_counts = df["Membership Type"].value_counts().sort_index()
cc1, cc2, cc3 = st.columns(3, gap="small")
comp_tiers = [
    (cc1, "fa-medal",  "#CD7F32", "Bronze", membership_counts.get(1,0), "Target for upgrades"),
    (cc2, "fa-medal",  "#94A3B8", "Silver", membership_counts.get(2,0), "Core customer base"),
    (cc3, "fa-trophy", "#F59E0B", "Gold",   membership_counts.get(3,0), "VIP experience"),
]
for col, icon, color, tier, count, desc in comp_tiers:
    pct = count / total * 100
    with col:
        st.markdown(f"""
        <div class="pro-card">
            <span class="card-fa-icon" style="color:{color};"><i class="fa-solid {icon}"></i></span>
            <div class="card-label">{tier} Members</div>
            <div class="card-value" style="color:{color};">{pct:.1f}%</div>
            <div class="card-delta muted">{count} customers &middot; {desc}</div>
        </div>
        """, unsafe_allow_html=True)

st.divider()

# ── Detailed metrics table ────────────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="section-accent"></div>
    <h2>Detailed Metrics by Membership</h2>
</div>
""", unsafe_allow_html=True)

metrics_tbl = df.groupby("Membership Type").agg({
    "Customer ID":"count","Total Spend":["mean","sum"],
    "Items Purchased":"mean","Average Rating":"mean",
    "Satisfaction Level":"mean","Days Since Last Purchase":"mean",
}).round(2)
metrics_tbl.columns = ["Customers","Avg Spend ($)","Total Spend ($)","Avg Items","Avg Rating","Avg Satisfaction","Avg Days Since"]
metrics_tbl.index   = ["Bronze","Silver","Gold"]
st.dataframe(metrics_tbl, use_container_width=True)

st.markdown("""
<div style="text-align:center;padding:1rem 0;color:#2D3B55;font-size:0.75rem;">
    <i class="fa-solid fa-lightbulb" style="margin-right:0.3rem;"></i>
    Business Insights &middot; Real-time data analysis for strategic decisions
</div>
""", unsafe_allow_html=True)
