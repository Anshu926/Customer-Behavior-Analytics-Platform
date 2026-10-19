"""
Satisfaction Predictor — 10k Consumer Behaviour Dataset
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from utils import (
    load_data, load_satisfaction_model, render_sidebar,
    get_chart_layout, prepare_input_for_prediction, predict_satisfaction,
    get_satisfaction_recommendation, get_feature_importance, FA_CDN,
    GENDER_MAPPING, INCOME_MAPPING, REGION_MAPPING,
    CATEGORY_MAPPING, PAYMENT_MAPPING, LOYALTY_MAPPING, SATISFACTION_MAPPING
)

st.set_page_config(page_title="Satisfaction Predictor", page_icon="🎯", layout="wide", initial_sidebar_state="expanded")
render_sidebar("prediction")

@st.cache_resource
def get_model():
    df = load_data()
    return load_satisfaction_model(df), df

model, df = get_model()

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown(f"""
{FA_CDN}
<div class="hero-banner" style="padding:1.6rem 2rem;">
    <div class="hero-tag"><i class="fa-solid fa-bullseye"></i> ML Prediction</div>
    <h1>Satisfaction Predictor</h1>
    <p class="hero-sub">Predict customer satisfaction level using our trained RandomForest model — trained on 10,000 consumer behaviour records.</p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── Input form ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="section-accent"></div>
    <h2>Customer Profile Input</h2>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="medium")

with col1:
    age          = st.slider("Customer Age", 18, 69, 35)
    gender       = st.selectbox("Gender", ["F", "M", "Other"],
                                format_func=lambda x: {"F": "Female (F)", "M": "Male (M)", "Other": "Other"}.get(x, x))
    income       = st.selectbox("Income Level", ["Low", "Medium", "High"])
    region       = st.selectbox("Region", ["Rural", "Semi-Urban", "Urban"])

with col2:
    category     = st.selectbox("Purchase Category", list(CATEGORY_MAPPING.keys()))
    avg_spend    = st.slider("Average Spend ($)", 10.0, 630.0, 200.0, step=5.0)
    payment      = st.selectbox("Payment Mode", list(PAYMENT_MAPPING.keys()))
    loyalty      = st.selectbox("Loyalty Member", ["No", "Yes"])

user_input = {
    "age": age, "gender": gender, "income": income,
    "region": region, "category": category, "avg_spend": avg_spend,
    "payment": payment, "loyalty": loyalty,
}

st.divider()

# ── Predict ───────────────────────────────────────────────────────────────────
if st.button("🎯  Predict Satisfaction", use_container_width=True):
    input_df = prepare_input_for_prediction(user_input)
    result   = predict_satisfaction(model, input_df)
    rec      = get_satisfaction_recommendation(result)

    SAT_COLORS = {0: "#EF4444", 1: "#F59E0B", 2: "#10B981"}
    color = SAT_COLORS[result["class"]]

    r1, r2, r3 = st.columns(3, gap="small")
    with r1:
        st.markdown(f"""
        <div class="result-card" style="background:{color}18;border:1px solid {color}44;">
            <div class="rc-label">Predicted Outcome</div>
            <div class="rc-value" style="color:{color};">{result['prediction']}</div>
        </div>""", unsafe_allow_html=True)
    with r2:
        st.markdown(f"""
        <div class="result-card" style="background:rgba(99,102,241,0.1);border:1px solid rgba(99,102,241,0.25);">
            <div class="rc-label">Model Confidence</div>
            <div class="rc-value" style="color:#A5B4FC;">{result['confidence']:.1f}%</div>
        </div>""", unsafe_allow_html=True)
    with r3:
        class_label = ["Low", "Moderate", "High"][result['class']]
        st.markdown(f"""
        <div class="result-card" style="background:rgba(16,185,129,0.08);border:1px solid rgba(16,185,129,0.25);">
            <div class="rc-label">Satisfaction Tier</div>
            <div class="rc-value" style="color:#34D399;">{class_label}</div>
        </div>""", unsafe_allow_html=True)

    st.divider()

    # Probability bars
    st.markdown("""
    <div class="section-header">
        <div class="section-accent"></div>
        <h2>Class Probabilities</h2>
    </div>""", unsafe_allow_html=True)

    labels = ["Low Satisfaction", "Moderate Satisfaction", "High Satisfaction"]
    colors = ["#EF4444", "#F59E0B", "#10B981"]
    fig_prob = go.Figure()
    for i, (lbl, prob, clr) in enumerate(zip(labels, result["probas"], colors)):
        fig_prob.add_trace(go.Bar(name=lbl, x=[prob * 100], y=[lbl],
                                  orientation="h", marker_color=clr,
                                  text=f"{prob*100:.1f}%", textposition="outside"))
    fig_prob.update_layout(**get_chart_layout(height=220, show_legend=False, title=""))
    st.plotly_chart(fig_prob, use_container_width=True)

    # Recommendation
    rec_color = SAT_COLORS[result["class"]]
    st.markdown(f"""
    <div style="background:rgba(17,24,39,0.9);border:1px solid {rec_color}44;border-left:4px solid {rec_color};
                border-radius:10px;padding:1.1rem 1.4rem;margin-top:0.5rem;">
        <div style="color:#94A3B8;font-size:0.7rem;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:0.45rem;">
            <i class="fa-solid fa-lightbulb" style="color:{rec_color};margin-right:0.4rem;"></i> Business Recommendation
        </div>
        <div style="color:#F1F5F9;font-size:0.92rem;line-height:1.6;">{rec}</div>
    </div>""", unsafe_allow_html=True)

    st.divider()

# ── Feature importance ────────────────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="section-accent"></div>
    <h2>Feature Importance</h2>
</div>""", unsafe_allow_html=True)

fi_df = get_feature_importance(model)
fig_fi = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                color="Importance", color_continuous_scale=["#1F2D45","#6366F1","#A5B4FC"],
                text=fi_df["Importance"].apply(lambda v: f"{v:.3f}"))
fig_fi.update_traces(textposition="outside")
fig_fi.update_layout(**get_chart_layout(height=350, show_legend=False, title="Feature Importance — RandomForest"))
fig_fi.update_coloraxes(showscale=False)
st.plotly_chart(fig_fi, use_container_width=True)
