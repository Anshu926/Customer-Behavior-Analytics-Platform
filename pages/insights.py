"""
Business Insights — 10k Consumer Behaviour Dataset
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
from utils import load_data, get_business_insights, ensure_features, render_sidebar, get_chart_layout, FA_CDN

st.set_page_config(page_title="Business Insights", page_icon="💡", layout="wide", initial_sidebar_state="expanded")
render_sidebar("insights")

@st.cache_data(ttl=1800)
def get_data_and_insights(v="v10k_prod"):
    df = load_data()
    df = ensure_features(df)
    return df, get_business_insights(df)

df, insights = get_data_and_insights()
df = ensure_features(df)
total = len(df)
mem_col = "Membership_Status" if "Membership_Status" in df.columns else "Loyalty_Member"

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown(f"""
{FA_CDN}
<div class="hero-banner" style="padding:1.6rem 2rem;">
    <div class="hero-tag"><i class="fa-solid fa-lightbulb"></i> Strategic Intelligence</div>
    <h1>Business Insights &amp; Recommendations</h1>
    <p class="hero-sub">Automated analytical intelligence surfacing revenue drivers, discount elasticity, recency churn hazards, and high-impact interventions from 10,000 customer records.</p>
</div>
""", unsafe_allow_html=True)

# ── Snapshot KPIs ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="section-accent"></div>
    <h2>Executive Insights at a Glance</h2>
</div>""", unsafe_allow_html=True)

k1, k2, k3, k4 = st.columns(4, gap="small")
snap = [
    (k1, "fa-face-smile",    "High Satisfaction",   f"{insights['high_sat_count']:,}",  "positive", f"{insights['high_sat_count']/total*100:.1f}% of base"),
    (k2, "fa-face-meh",      "Moderate Tier",       f"{insights['mod_sat_count']:,}",   "neutral",  f"{insights['mod_sat_count']/total*100:.1f}% of base"),
    (k3, "fa-face-frown",    "Low / At Risk",       f"{insights['low_sat_count']:,}",   "neutral",  f"{insights['low_sat_count']/total*100:.1f}% churn risk"),
    (k4, "fa-basket-shopping","Avg Basket Size",    f"{insights.get('avg_items', 4.3):.1f} items", "info", "Items per transaction"),
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
        <div class="ip-num">01 — TOP PRODUCT CATEGORY</div>
        <h4>🏆 {insights['top_category']}</h4>
        <p style="color:#94A3B8;font-size:0.84rem;line-height:1.6;">
            <strong>{insights['top_category']}</strong> drives the highest revenue density with an average spend of
            <strong style="color:#A5B4FC;">${insights['top_category_spend']:.2f}</strong> per customer.
            Recommend boosting supplier partnerships and cross-selling higher-margin accessories in this category.
        </p>
        <div class="badge badge-indigo" style="font-size:0.7rem;">Revenue Benchmark</div>
    </div>""", unsafe_allow_html=True)

with i1r:
    cat_spend = df.groupby("Purchase_Category")["Avg_Spend"].mean().sort_values(ascending=False)
    fig1 = px.bar(x=cat_spend.index, y=cat_spend.values, color=cat_spend.index,
                  color_discrete_sequence=["#6366F1","#10B981","#F59E0B","#EF4444","#8B5CF6","#06B6D4"],
                  text=[f"${v:.0f}" for v in cat_spend.values],
                  labels={"x": "Category", "y": "Avg Spend ($)"})
    fig1.update_traces(textposition="outside", marker=dict(line=dict(width=0)), width=0.5)
    fig1.update_layout(**get_chart_layout(height=300, show_legend=False))
    st.plotly_chart(fig1, use_container_width=True)

st.divider()

# ── Insight 2: Membership & Loyalty Impact ────────────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="section-accent"></div>
    <h2>Insight 2 &middot; Membership Program Value &amp; ROI</h2>
</div>""", unsafe_allow_html=True)

i2l, i2r = st.columns([1, 2], gap="small")
loyalty_spend = df.groupby(mem_col)["Avg_Spend"].mean()
loyalty_sat   = df.groupby(mem_col)["Satisfaction_Score"].mean()
loyalty_items = df.groupby(mem_col)["Items_Purchased"].mean() if "Items_Purchased" in df.columns else None

with i2l:
    loy_pct = insights["loyalty_pct"]
    st.markdown(f"""
    <div class="insight-panel">
        <div class="ip-num">02 — MEMBERSHIP VALUE</div>
        <h4>❤️ {loy_pct:.1f}% Enrolled Members</h4>
        <p style="color:#94A3B8;font-size:0.84rem;line-height:1.6;">
            Enrolled members spend <strong style="color:#34D399;">${loyalty_spend.get('Yes',0):.2f}</strong> vs
            <strong style="color:#FCA5A5;">${loyalty_spend.get('No',0):.2f}</strong> for non-members.
            Members also maintain an average satisfaction of <strong style="color:#A5B4FC;">{loyalty_sat.get('Yes',0):.2f} / 5</strong>.
        </p>
        <div class="badge badge-green" style="font-size:0.7rem;">High Lifetime Value</div>
    </div>""", unsafe_allow_html=True)

with i2r:
    loy_df = pd.DataFrame({
        "Status": ["Non-Member", "Enrolled Member"],
        "Avg Spend": [loyalty_spend.get("No", 0), loyalty_spend.get("Yes", 0)],
        "Satisfaction Score": [loyalty_sat.get("No", 0), loyalty_sat.get("Yes", 0)],
    })
    fig2 = px.bar(loy_df, x="Status", y="Avg Spend",
                  color="Status", color_discrete_map={"Non-Member":"#6366F1","Enrolled Member":"#10B981"},
                  text=loy_df["Avg Spend"].apply(lambda v: f"${v:.1f}"),
                  labels={"Status": "Customer Status", "Avg Spend": "Average Spend ($)"})
    fig2.update_traces(textposition="outside", width=0.35)
    fig2.update_layout(**get_chart_layout(height=300, show_legend=False, title="Spend: Members vs Non-Members"))
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ── Insight 3: Discount Elasticity ────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="section-accent"></div>
    <h2>Insight 3 &middot; Discount Engagement &amp; Basket Dynamics</h2>
</div>""", unsafe_allow_html=True)

i3l, i3r = st.columns([1, 2], gap="small")
if "Discount_Applied" not in df.columns:
    df["Discount_Applied"] = False
if "Items_Purchased" not in df.columns:
    df["Items_Purchased"] = 4

disc_grp = df.groupby("Discount_Applied").agg(
    Avg_Spend=("Avg_Spend", "mean"),
    Avg_Items=("Items_Purchased", "mean"),
    Avg_Sat=("Satisfaction_Score", "mean")
).reset_index()
disc_grp["Type"] = disc_grp["Discount_Applied"].map({True: "With Discount", False: "Full Price"})

with i3l:
    disc_rate = insights.get("discount_pct", 53.0)
    st.markdown(f"""
    <div class="insight-panel">
        <div class="ip-num">03 — DISCOUNT IMPACT</div>
        <h4>🏷️ {disc_rate:.1f}% Discount Penetration</h4>
        <p style="color:#94A3B8;font-size:0.84rem;line-height:1.6;">
            Discounted orders average <strong style="color:#34D399;">{disc_grp[disc_grp['Discount_Applied']==True]['Avg_Items'].values[0]:.1f} items</strong> per order
            at an average spend of <strong style="color:#A5B4FC;">${disc_grp[disc_grp['Discount_Applied']==True]['Avg_Spend'].values[0]:.0f}</strong>.
            Strategic flash discounts incentivize larger basket checkouts without sacrificing margin.
        </p>
        <div class="badge badge-amber" style="font-size:0.7rem;">Basket Growth Driver</div>
    </div>""", unsafe_allow_html=True)

with i3r:
    fig3 = px.bar(disc_grp, x="Type", y="Avg_Spend",
                  color="Type", color_discrete_map={"With Discount": "#10B981", "Full Price": "#6366F1"},
                  text=disc_grp["Avg_Spend"].apply(lambda v: f"${v:.1f}"),
                  labels={"Type": "Pricing Strategy", "Avg_Spend": "Average Spend ($)"})
    fig3.update_traces(textposition="outside", width=0.35)
    fig3.update_layout(**get_chart_layout(height=300, show_legend=False, title="Average Spend by Discount Status"))
    st.plotly_chart(fig3, use_container_width=True)

st.divider()

# ── Insight 4: Recency & Churn Risk ───────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="section-accent"></div>
    <h2>Insight 4 &middot; Purchase Recency &amp; Churn Risk</h2>
</div>""", unsafe_allow_html=True)

df["Recency_Group"] = pd.cut(
    df["Days_Since_Last_Purchase"],
    bins=[-1, 15, 30, 60, 100],
    labels=["< 15 Days", "15–30 Days", "31–60 Days", "60+ Days"]
)
rec_df = df.groupby("Recency_Group", observed=True).agg(
    Avg_Sat=("Satisfaction_Score", "mean"),
    Avg_Spend=("Avg_Spend", "mean"),
    Count=("Customer_ID", "count")
).reset_index()

i4l, i4r = st.columns([1, 2], gap="small")
with i4l:
    dormant_count = rec_df[rec_df["Recency_Group"] == "60+ Days"]["Count"].values[0] if len(rec_df) else 0
    st.markdown(f"""
    <div class="insight-panel">
        <div class="ip-num">04 — CHURN &amp; RECENCY</div>
        <h4>⚠️ {dormant_count:,} Dormant Customers</h4>
        <p style="color:#94A3B8;font-size:0.84rem;line-height:1.6;">
            Customers whose last purchase exceeds 60 days exhibit significantly lower satisfaction scores
            (<strong style="color:#EF4444;">{rec_df[rec_df['Recency_Group']=='60+ Days']['Avg_Sat'].values[0]:.2f} / 5</strong>).
            Setting an automated re-engagement trigger at <strong>Day 25</strong> prevents dormant churn.
        </p>
        <div class="badge badge-red" style="font-size:0.7rem;">High Attrition Risk</div>
    </div>""", unsafe_allow_html=True)

with i4r:
    fig4 = px.line(rec_df, x="Recency_Group", y="Avg_Sat",
                   markers=True, line_shape="spline",
                   labels={"Recency_Group": "Days Since Last Purchase", "Avg_Sat": "Avg Satisfaction Score"})
    fig4.update_traces(line_color="#EF4444", line_width=3, marker=dict(size=9, color="#F59E0B"))
    fig4.update_layout(**get_chart_layout(height=300, show_legend=False, title="Satisfaction Score vs Purchase Recency"))
    st.plotly_chart(fig4, use_container_width=True)

st.divider()

# ── Insight 5: Payment Mode & Satisfaction ────────────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="section-accent"></div>
    <h2>Insight 5 &middot; Payment Mode Adoption</h2>
</div>""", unsafe_allow_html=True)

pay_sat = df.groupby("Payment_Mode")["Satisfaction_Score"].mean().sort_values(ascending=False)
fig5 = px.bar(x=pay_sat.index, y=pay_sat.values,
              color=pay_sat.index,
              color_discrete_sequence=["#6366F1","#10B981","#F59E0B","#EF4444","#8B5CF6"],
              text=[f"{v:.2f}" for v in pay_sat.values],
              labels={"x": "Payment Mode", "y": "Avg Satisfaction Score"})
fig5.update_traces(textposition="outside", marker=dict(line=dict(width=0)), width=0.5)
fig5.update_layout(**get_chart_layout(height=340, show_legend=False, title="Customer Satisfaction by Payment Method"))
st.plotly_chart(fig5, use_container_width=True)

st.divider()

# ── Strategic Recommendations ─────────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="section-accent"></div>
    <h2>Actionable Strategic Recommendations</h2>
</div>""", unsafe_allow_html=True)

recs = [
    ("fa-trophy",       "#10B981", "Scale Top Performing Category",
     f"Accelerate marketing and inventory expansion for <strong>{insights['top_category']}</strong>, which yields the top average spend of ${insights['top_category_spend']:.2f}.",
     "Revenue Growth", "High Impact"),
    ("fa-heart",        "#6366F1", "Convert Non-Members via Targeted Perks",
     f"With {insights['loyalty_pct']:.1f}% membership enrollment, non-members represent an immediate conversion opportunity. Offer free shipping and welcome vouchers on first signup.",
     "Customer Lifetime Value", "High Impact"),
    ("fa-bell",         "#F59E0B", "Automate Day-25 Win-Back Sequences",
     "Satisfaction drops precipitously after 30+ days of dormancy. Implement automated email/SMS reminders with tailored recommendations before Day 30.",
     "Churn Prevention", "Critical"),
    ("fa-tags",         "#EC4899", "Optimize Basket-Building Discounts",
     f"Discounts are applied on {insights.get('discount_pct', 53.0):.1f}% of transactions. Bundle high-margin items with multi-buy thresholds to maximize average order value.",
     "Margin Optimization", "Medium Impact"),
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
