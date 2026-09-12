"""
Customer Segmentation — KMeans Clustering Analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils import (
    load_data, load_segmentation_model, get_cluster_labels,
    render_sidebar, get_chart_layout, FA_CDN,
    MEMBERSHIP_REVERSE,
)

st.set_page_config(page_title="Customer Segmentation", page_icon="🔵", layout="wide", initial_sidebar_state="expanded")
render_sidebar("segmentation")

@st.cache_data
def get_cached_data():
    return load_data()

df = get_cached_data()
kmeans_model, scaler = load_segmentation_model(df)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown(f"""
{FA_CDN}
<div class="hero-banner" style="padding:1.6rem 2rem;">
    <div class="hero-tag"><i class="fa-solid fa-layer-group"></i> KMeans Clustering</div>
    <h1>Customer Segmentation</h1>
    <p class="hero-sub">Discover behavioral personas through intelligent clustering — understand spending patterns and tailor strategy to each segment.</p>
</div>
""", unsafe_allow_html=True)

# ── Model info cards ──────────────────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="section-accent"></div>
    <h2>Segmentation Model</h2>
</div>
""", unsafe_allow_html=True)

mi1, mi2, mi3, mi4 = st.columns(4, gap="small")
info_cards = [
    (mi1, "fa-circle-nodes",  "Clustering Type",  "K-Means",      "info",     "Unsupervised ML"),
    (mi2, "fa-object-group",  "Clusters",         "4 Segments",   "info",     "Customer groups"),
    (mi3, "fa-ruler-combined","Features Used",    "6 Features",   "info",     "Spend, age, items…"),
    (mi4, "fa-users",         "Total Customers",  f"{len(df):,}", "positive", "Segmented & analysed"),
]
for col, icon, label, value, cls, sub in info_cards:
    with col:
        st.markdown(f"""
        <div class="pro-card">
            <span class="card-fa-icon"><i class="fa-solid {icon}"></i></span>
            <div class="card-label">{label}</div>
            <div class="card-value" style="font-size:1.2rem;">{value}</div>
            <div class="card-delta {cls}">{sub}</div>
        </div>
        """, unsafe_allow_html=True)

st.divider()

# ── Assign segments ───────────────────────────────────────────────────────────
segment_labels = get_cluster_labels()

@st.cache_data
def assign_segments(_df, _kmeans, _scaler, _labels):
    df2 = _df.copy()
    df2["Segment"] = df2.apply(lambda row: _labels[_kmeans.predict(
        _scaler.transform([[row["Age"], row["Membership Type"], row["Total Spend"],
                            row["Items Purchased"], row["Average Rating"],
                            row["Days Since Last Purchase"]]])
    )[0]], axis=1)
    return df2

df = assign_segments(df, kmeans_model, scaler, segment_labels)
segment_counts = df["Segment"].value_counts()

SEG_COLORS = {
    "Premium Customers":    "#F59E0B",
    "Budget Customers":     "#94A3B8",
    "Frequent Buyers":      "#6366F1",
    "Low Engagement Users": "#EF4444",
}
SEG_FA_ICONS = {
    "Premium Customers":    "fa-crown",
    "Budget Customers":     "fa-briefcase",
    "Frequent Buyers":      "fa-rotate",
    "Low Engagement Users": "fa-moon",
}

# ── Segment distribution charts ───────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="section-accent"></div>
    <h2>Segment Distribution</h2>
</div>
""", unsafe_allow_html=True)

sd1, sd2 = st.columns(2, gap="small")

with sd1:
    fig_pie = px.pie(
        values=segment_counts.values,
        names=segment_counts.index,
        color=segment_counts.index,
        color_discrete_map=SEG_COLORS,
        hole=0.45,
    )
    fig_pie.update_traces(
        textposition="outside", textinfo="percent+label",
        textfont=dict(size=11, color="#94A3B8"),
        marker=dict(line=dict(color="#111827", width=2)),
    )
    pie_lay = get_chart_layout(height=380, show_legend=True, title="Segment Breakdown")
    pie_lay["xaxis"] = dict(visible=False)
    pie_lay["yaxis"] = dict(visible=False)
    fig_pie.update_layout(**pie_lay)
    st.plotly_chart(fig_pie, use_container_width=True)

with sd2:
    seg_df = pd.DataFrame({
        "Segment": segment_counts.index,
        "Count":   segment_counts.values,
    })
    fig_bar = px.bar(
        seg_df, x="Segment", y="Count",
        color="Segment", color_discrete_map=SEG_COLORS,
        text=seg_df["Count"],
    )
    fig_bar.update_traces(textposition="outside", marker=dict(line=dict(width=0)), width=0.5)
    fig_bar.update_layout(**get_chart_layout(height=380, show_legend=False, title="Customer Count per Segment"))
    st.plotly_chart(fig_bar, use_container_width=True)

st.divider()

# ── Segment profiles ──────────────────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="section-accent"></div>
    <h2>Segment Profiles</h2>
</div>
""", unsafe_allow_html=True)

profiles = {}
for seg in segment_labels.values():
    seg_df = df[df["Segment"] == seg]
    profiles[seg] = {
        "size":       len(seg_df),
        "avg_spend":  seg_df["Total Spend"].mean(),
        "avg_items":  seg_df["Items Purchased"].mean(),
        "avg_rating": seg_df["Average Rating"].mean(),
        "membership": seg_df["Membership Type"].map(MEMBERSHIP_REVERSE).mode()[0] if len(seg_df) > 0 else "N/A",
    }

segs = list(segment_labels.values())
sp1, sp2 = st.columns(2, gap="small")

for i, seg_name in enumerate(segs):
    col   = sp1 if i % 2 == 0 else sp2
    info  = profiles[seg_name]
    color = SEG_COLORS.get(seg_name, "#94A3B8")
    icon  = SEG_FA_ICONS.get(seg_name, "fa-circle")
    pct   = info["size"] / len(df) * 100

    with col:
        st.markdown(f"""
        <div class="seg-card">
            <div class="sc-header">
                <div class="sc-stripe" style="background:{color};"></div>
                <div class="sc-icon-wrap" style="background:{color}18;border:1px solid {color}33;">
                    <i class="fa-solid {icon}" style="color:{color};"></i>
                </div>
                <div>
                    <div class="sc-title">{seg_name}</div>
                    <span class="badge" style="background:{color}18;color:{color};border:1px solid {color}33;margin-top:3px;display:inline-block;">
                        {info['size']} customers &middot; {pct:.1f}%
                    </span>
                </div>
            </div>
            <div class="sc-stats">
                <div>
                    <div class="sc-stat-label"><i class="fa-solid fa-dollar-sign" style="width:12px;"></i> Avg Spend</div>
                    <div class="sc-stat-value">${info['avg_spend']:.0f}</div>
                </div>
                <div>
                    <div class="sc-stat-label"><i class="fa-solid fa-box" style="width:12px;"></i> Avg Items</div>
                    <div class="sc-stat-value">{info['avg_items']:.1f}</div>
                </div>
                <div>
                    <div class="sc-stat-label"><i class="fa-solid fa-star" style="width:12px;"></i> Avg Rating</div>
                    <div class="sc-stat-value">{info['avg_rating']:.2f} / 5</div>
                </div>
                <div>
                    <div class="sc-stat-label"><i class="fa-solid fa-id-badge" style="width:12px;"></i> Membership</div>
                    <div class="sc-stat-value">{info['membership']}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.divider()

# ── Radar chart ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="section-accent"></div>
    <h2>Segment Comparison — Radar</h2>
</div>
""", unsafe_allow_html=True)

radar_features = ["Total Spend", "Items Purchased", "Average Rating", "Days Since Last Purchase", "Age"]
radar_data     = df.groupby("Segment")[radar_features].mean()
fig_radar      = go.Figure()

for seg in radar_data.index:
    vals = radar_data.loc[seg].tolist()
    maxv = radar_data.max()
    norm = [v / mx if mx else 0 for v, mx in zip(vals, maxv.tolist())]
    norm.append(norm[0])
    color = SEG_COLORS.get(seg, "#94A3B8")
    # Build rgba fill
    r,g,b = int(color[1:3],16), int(color[3:5],16), int(color[5:7],16)
    fig_radar.add_trace(go.Scatterpolar(
        r=norm, theta=radar_features + [radar_features[0]],
        fill="toself", name=seg,
        line=dict(color=color, width=2),
        fillcolor=f"rgba({r},{g},{b},0.07)",
    ))

radar_lay = get_chart_layout(height=440, show_legend=True, title="Normalised Segment Profiles")
radar_lay["polar"] = dict(
    bgcolor="#111827",
    radialaxis=dict(visible=True, range=[0,1], gridcolor="#1F2D45", tickfont=dict(size=9,color="#64748B")),
    angularaxis=dict(gridcolor="#1F2D45", tickfont=dict(size=10,color="#94A3B8")),
)
fig_radar.update_layout(**radar_lay)
st.plotly_chart(fig_radar, use_container_width=True)

st.divider()

# ── Recommended actions ───────────────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="section-accent"></div>
    <h2>Recommended Actions by Segment</h2>
</div>
""", unsafe_allow_html=True)

actions = {
    "Premium Customers":    {"icon":"fa-crown",    "color":"#F59E0B","items":["VIP treatment with exclusive offers","Dedicated account manager","Early access to new products","Premium support priority"]},
    "Budget Customers":     {"icon":"fa-briefcase","color":"#94A3B8","items":["Value-focused promotions","Bulk discount programmes","Price comparison guidance","Cost-saving recommendations"]},
    "Frequent Buyers":      {"icon":"fa-rotate",   "color":"#6366F1","items":["Loyalty rewards programme","Personalised product recommendations","Exclusive member-only events","Referral bonus incentives"]},
    "Low Engagement Users": {"icon":"fa-moon",     "color":"#EF4444","items":["Targeted re-engagement campaigns","Special comeback offers","Personalised outreach emails","Win-back promotions"]},
}

ac1, ac2 = st.columns(2, gap="small")
for i, (seg_name, action) in enumerate(actions.items()):
    col = ac1 if i % 2 == 0 else ac2
    items_html = "".join(
        f'<div style="color:#94A3B8;font-size:0.81rem;padding:0.22rem 0;display:flex;align-items:center;gap:0.4rem;">'
        f'<i class="fa-solid fa-chevron-right" style="font-size:0.55rem;color:{action["color"]};"></i>{item}</div>'
        for item in action["items"]
    )
    with col:
        st.markdown(f"""
        <div class="rec-card" style="border-top-color:{action['color']};">
            <div class="rc-title">
                <i class="fa-solid {action['icon']}" style="color:{action['color']};margin-right:0.4rem;"></i>
                {seg_name}
            </div>
            {items_html}
        </div>
        """, unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center;padding:1rem 0;color:#2D3B55;font-size:0.75rem;">
    <i class="fa-solid fa-circle-nodes" style="margin-right:0.3rem;"></i>
    Segmentation &middot; Powered by scikit-learn KMeans
</div>
""", unsafe_allow_html=True)
