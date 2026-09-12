"""
Satisfaction Predictor — RandomForest ML Model
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils import (
    load_data, load_satisfaction_model,
    prepare_input_for_prediction, predict_satisfaction,
    get_satisfaction_recommendation, get_feature_importance,
    render_sidebar, get_chart_layout, FA_CDN,
    MEMBERSHIP_MAPPING, GENDER_MAPPING,
)

st.set_page_config(page_title="Satisfaction Predictor", page_icon="🎯", layout="wide", initial_sidebar_state="expanded")
render_sidebar("prediction")

@st.cache_resource
def load_model_and_data():
    df    = load_data()
    model = load_satisfaction_model(df)
    return df, model

df, model = load_model_and_data()

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown(f"""
{FA_CDN}
<div class="hero-banner" style="padding:1.6rem 2rem;">
    <div class="hero-tag"><i class="fa-solid fa-bullseye"></i> ML Prediction Engine</div>
    <h1>Satisfaction Predictor</h1>
    <p class="hero-sub">Forecast customer satisfaction using a trained RandomForest classifier — with confidence scores and actionable recommendations.</p>
</div>
""", unsafe_allow_html=True)

# ── Model info cards ──────────────────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="section-accent"></div>
    <h2>Model Information</h2>
</div>
""", unsafe_allow_html=True)

mi1, mi2, mi3, mi4 = st.columns(4, gap="small")
model_cards = [
    (mi1, "fa-tree",         "Model Type",     "Random Forest",  "Classification"),
    (mi2, "fa-sitemap",      "Estimators",     "100",            "Decision Trees"),
    (mi3, "fa-list-ol",      "Output Classes", "3 classes",      "Satisfaction Levels"),
    (mi4, "fa-circle-check", "Status",         "Ready",          "Production"),
]
for col, icon, label, value, sub in model_cards:
    with col:
        st.markdown(f"""
        <div class="pro-card">
            <span class="card-fa-icon"><i class="fa-solid {icon}"></i></span>
            <div class="card-label">{label}</div>
            <div class="card-value" style="font-size:1.2rem;">{value}</div>
            <div class="card-delta info">{sub}</div>
        </div>
        """, unsafe_allow_html=True)

st.divider()

# ── Input form ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="section-accent"></div>
    <h2>Customer Profile</h2>
</div>
""", unsafe_allow_html=True)

with st.container(border=True):
    f1, f2, f3 = st.columns(3, gap="medium")

    with f1:
        st.markdown("**Demographics**")
        gender    = st.selectbox("Gender", ["Female", "Male"])
        age       = st.slider("Age", 18, 80, 30)
        membership = st.selectbox("Membership Type", ["Bronze", "Silver", "Gold"])

    with f2:
        st.markdown("**Purchase Behaviour**")
        total_spend      = st.slider("Total Spend ($)", 100.0, 2000.0, 800.0, 50.0)
        items_purchased  = st.slider("Items Purchased", 1, 50, 10)
        avg_rating       = st.slider("Average Rating", 1.0, 5.0, 4.0, 0.1)

    with f3:
        st.markdown("**Context**")
        discount  = st.checkbox("Discount Applied", value=False)
        days_since = st.slider("Days Since Last Purchase", 1, 365, 30)
        city_tier = st.selectbox("City Tier", [1,2,3],
                                 format_func=lambda x: f"Tier {x} — {'Low' if x==1 else 'Medium' if x==2 else 'High'}")


# ── Buttons ───────────────────────────────────────────────────────────────────
btn1, btn2, _s = st.columns([1.5, 1.3, 3])
with btn1:
    predict_btn = st.button("🎯 Predict Satisfaction", type="primary", use_container_width=True)
with btn2:
    reset_btn = st.button("↺ Reset Form", type="secondary", use_container_width=True)

if reset_btn:
    st.rerun()

# ── Results ───────────────────────────────────────────────────────────────────
if predict_btn:
    user_input = {
        "gender":gender,"age":age,"membership":membership,
        "total_spend":total_spend,"items_purchased":items_purchased,
        "avg_rating":avg_rating,"discount":discount,
        "days_since_purchase":days_since,"city_tier":city_tier,
    }
    input_df       = prepare_input_for_prediction(user_input)
    result         = predict_satisfaction(model, input_df)
    recommendation = get_satisfaction_recommendation(result)

    SAT_STYLES = {
        "Satisfied":   {"bg":"rgba(16,185,129,0.12)","border":"#10B981","text":"#34D399","fa_icon":"fa-face-smile"},
        "Neutral":     {"bg":"rgba(245,158,11,0.12)", "border":"#F59E0B","text":"#FCD34D","fa_icon":"fa-face-meh"},
        "Unsatisfied": {"bg":"rgba(239,68,68,0.12)",  "border":"#EF4444","text":"#FCA5A5","fa_icon":"fa-face-frown"},
    }
    style = SAT_STYLES[result["prediction"]]

    st.divider()
    st.markdown("""
    <div class="section-header">
        <div class="section-accent"></div>
        <h2>Prediction Results</h2>
    </div>
    """, unsafe_allow_html=True)

    r1, r2, r3 = st.columns(3, gap="small")

    with r1:
        st.markdown(f"""
        <div class="result-card" style="
            background:{style['bg']};
            border:2px solid {style['border']};
            animation:glowPulse 2.5s ease infinite;
        ">
            <div class="rc-label" style="color:{style['text']};">Predicted Satisfaction</div>
            <div style="font-size:2.2rem;color:{style['text']};margin:0.4rem 0;">
                <i class="fa-solid {style['fa_icon']}"></i>
            </div>
            <div class="rc-value" style="color:{style['text']};">{result['prediction']}</div>
        </div>
        """, unsafe_allow_html=True)

    with r2:
        conf = result["confidence"]
        gauge_color = "#10B981" if conf >= 75 else "#F59E0B" if conf >= 50 else "#EF4444"
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=conf,
            number=dict(suffix="%", font=dict(size=28,color=gauge_color)),
            gauge=dict(
                axis=dict(range=[0,100],tickwidth=1,tickcolor="#1F2D45",tickfont=dict(color="#64748B",size=9)),
                bar=dict(color=gauge_color,thickness=0.25),
                bgcolor="#111827", borderwidth=1, bordercolor="#1F2D45",
                steps=[dict(range=[0,50],color="#1A1F2E"),dict(range=[50,75],color="#1E2A1E"),dict(range=[75,100],color="#1A2E1A")],
                threshold=dict(line=dict(color=gauge_color,width=3),thickness=0.75,value=conf),
            ),
            title=dict(text="Confidence Score",font=dict(size=13,color="#94A3B8")),
        ))
        fig_gauge.update_layout(paper_bgcolor="#111827",font=dict(color="#94A3B8"),height=200,margin=dict(l=20,r=20,t=40,b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with r3:
        icon_map = {"Satisfied":"fa-circle-check","Neutral":"fa-triangle-exclamation","Unsatisfied":"fa-circle-xmark"}
        st.markdown(f"""
        <div style="background:#111827;border:1px solid #1F2D45;border-radius:14px;padding:1.4rem;min-height:170px;display:flex;flex-direction:column;justify-content:center;">
            <div style="color:#64748B;font-size:0.68rem;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:0.6rem;">
                <i class="fa-solid fa-comment-dots" style="margin-right:0.3rem;"></i> Business Recommendation
            </div>
            <div style="color:#F1F5F9;font-size:0.9rem;line-height:1.65;">
                <i class="fa-solid {icon_map.get(result['prediction'],'fa-info-circle')}" style="color:{style['text']};margin-right:0.4rem;"></i>
                {recommendation}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Probability breakdown
    st.markdown("<br>", unsafe_allow_html=True)
    labels_order = ["Unsatisfied","Neutral","Satisfied"]
    proba_colors = {"Unsatisfied":"#EF4444","Neutral":"#F59E0B","Satisfied":"#10B981"}
    for i, lbl in enumerate(labels_order):
        pct = result["probas"][i] * 100
        col_color = proba_colors[lbl]
        st.markdown(f"""
        <div style="margin-bottom:0.55rem;">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.2rem;">
                <span style="color:#94A3B8;font-size:0.8rem;font-weight:500;">{lbl}</span>
                <span style="color:{col_color};font-size:0.8rem;font-weight:600;">{pct:.1f}%</span>
            </div>
            <div style="background:#1F2D45;border-radius:4px;height:6px;">
                <div style="background:{col_color};width:{pct:.1f}%;height:6px;border-radius:4px;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Profile summary
    st.divider()
    st.markdown("""
    <div class="section-header">
        <div class="section-accent"></div>
        <h2>Customer Profile Summary</h2>
    </div>
    """, unsafe_allow_html=True)

    ps1, ps2 = st.columns(2, gap="small")
    with ps1:
        st.markdown(f"""
        <div class="insight-panel">
            <div class="ip-num"><i class="fa-solid fa-id-card" style="margin-right:0.3rem;"></i>Demographics</div>
            <h4>Personal Details</h4>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.4rem 1rem;">
                <div><div style="color:#64748B;font-size:0.7rem;">Gender</div><div style="color:#F1F5F9;font-size:0.88rem;font-weight:500;">{gender}</div></div>
                <div><div style="color:#64748B;font-size:0.7rem;">Age</div><div style="color:#F1F5F9;font-size:0.88rem;font-weight:500;">{age} yrs</div></div>
                <div><div style="color:#64748B;font-size:0.7rem;">City Tier</div><div style="color:#F1F5F9;font-size:0.88rem;font-weight:500;">{'Low' if city_tier==1 else 'Medium' if city_tier==2 else 'High'}</div></div>
                <div><div style="color:#64748B;font-size:0.7rem;">Days Since Purchase</div><div style="color:#F1F5F9;font-size:0.88rem;font-weight:500;">{days_since} days</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with ps2:
        st.markdown(f"""
        <div class="insight-panel">
            <div class="ip-num"><i class="fa-solid fa-bag-shopping" style="margin-right:0.3rem;"></i>Purchase Behaviour</div>
            <h4>Shopping Pattern</h4>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.4rem 1rem;">
                <div><div style="color:#64748B;font-size:0.7rem;">Membership</div><div style="color:#F1F5F9;font-size:0.88rem;font-weight:500;">{membership}</div></div>
                <div><div style="color:#64748B;font-size:0.7rem;">Total Spend</div><div style="color:#F1F5F9;font-size:0.88rem;font-weight:500;">${total_spend:.2f}</div></div>
                <div><div style="color:#64748B;font-size:0.7rem;">Items Bought</div><div style="color:#F1F5F9;font-size:0.88rem;font-weight:500;">{items_purchased}</div></div>
                <div><div style="color:#64748B;font-size:0.7rem;">Avg Rating</div><div style="color:#F1F5F9;font-size:0.88rem;font-weight:500;">{avg_rating}/5.0</div></div>
                <div style="grid-column:span 2;"><div style="color:#64748B;font-size:0.7rem;">Discount Applied</div><div style="color:#F1F5F9;font-size:0.88rem;font-weight:500;">{'Yes' if discount else 'No'}</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.divider()

# ── Feature importance ────────────────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="section-accent"></div>
    <h2>Feature Importance</h2>
</div>
""", unsafe_allow_html=True)
st.markdown('<p style="margin-bottom:0.8rem;">Which factors influence the satisfaction prediction the most?</p>', unsafe_allow_html=True)

importance_df = get_feature_importance(model).head(9)
n = len(importance_df)
colours = [f"rgba(99,102,241,{0.95 - i*(0.55/n):.2f})" for i in range(n)]

fig_imp = go.Figure(go.Bar(
    y=importance_df["Feature"], x=importance_df["Importance"],
    orientation="h", marker=dict(color=colours, line=dict(width=0)),
    text=[f"{v:.3f}" for v in importance_df["Importance"]],
    textposition="outside", textfont=dict(size=10, color="#94A3B8"),
))
imp_lay = get_chart_layout(height=380, show_legend=False, title="Top Features Influencing Satisfaction")
imp_lay["yaxis"]["autorange"] = "reversed"
imp_lay["xaxis"]["title"] = "Importance Score"
fig_imp.update_layout(**imp_lay)
st.plotly_chart(fig_imp, use_container_width=True)

# ── How to use ────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div class="section-header">
    <div class="section-accent"></div>
    <h2>How to Use</h2>
</div>
""", unsafe_allow_html=True)

steps = [
    ("fa-keyboard",     "1", "Fill in Customer Details", "Enter the customer's demographic and purchase information using the form above."),
    ("fa-wand-magic-sparkles","2","Click Predict",        "The RandomForest model will instantly compute the satisfaction prediction."),
    ("fa-chart-pie",    "3", "Review Results",           "See the prediction label, confidence gauge, probability bars, and recommendation."),
    ("fa-bolt",         "4", "Take Action",              "Use the business recommendation to improve customer experience and retention."),
]
hw1, hw2 = st.columns(2, gap="small")
for i, (icon, num, title, desc) in enumerate(steps):
    col = hw1 if i % 2 == 0 else hw2
    with col:
        st.markdown(f"""
        <div class="insight-panel" style="margin-bottom:0.65rem;">
            <div class="ip-num"><i class="fa-solid {icon}" style="margin-right:0.3rem;"></i> Step {num}</div>
            <h4>{title}</h4>
            <p style="margin:0;">{desc}</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center;padding:1rem 0;color:#2D3B55;font-size:0.75rem;">
    <i class="fa-solid fa-tree" style="margin-right:0.3rem;"></i>
    Satisfaction Predictor &middot; Powered by scikit-learn RandomForest
</div>
""", unsafe_allow_html=True)
