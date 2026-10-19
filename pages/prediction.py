"""
Satisfaction Predictor — ML Model Inference
Predicts Satisfaction_Score & Satisfaction Class using:
Gender, Age, Income_Level, Region, Avg_Spend, Membership.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from utils import (
    load_data, load_satisfaction_model, render_sidebar,
    get_chart_layout, prepare_input_for_prediction, predict_satisfaction,
    get_satisfaction_recommendation, get_feature_importance, FA_CDN,
    GENDER_MAPPING, INCOME_MAPPING, REGION_MAPPING,
    MEMBERSHIP_MAPPING, SATISFACTION_MAPPING
)

st.set_page_config(page_title="Satisfaction Predictor", page_icon="🎯", layout="wide", initial_sidebar_state="expanded")
render_sidebar("prediction")

@st.cache_resource
def get_model():
    df = load_data()
    return load_satisfaction_model(df), df

model_bundle, df = get_model()

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown(f"""
{FA_CDN}
<div class="hero-banner" style="padding:1.6rem 2rem;">
    <div class="hero-tag"><i class="fa-solid fa-bullseye"></i> ML Satisfaction Prediction</div>
    <h1>Satisfaction Predictor</h1>
    <p class="hero-sub">Predict customer <strong>Satisfaction_Score (1.0 – 5.0)</strong> and satisfaction tier using machine learning trained on 10,000 customer records.</p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── Input form ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="section-accent"></div>
    <h2>Customer Profile Inputs</h2>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="medium")

with col1:
    age        = st.slider("Age", 18, 69, 35)
    gender     = st.selectbox("Gender", ["F", "M", "Other"],
                              format_func=lambda x: {"F": "Female (F)", "M": "Male (M)", "Other": "Other"}.get(x, x))
    income     = st.selectbox("Income Level", ["Low", "Medium", "High"], index=1)

with col2:
    region     = st.selectbox("Region", ["Rural", "Semi-Urban", "Urban"], index=2)
    avg_spend  = st.slider("Average Spend ($)", 10.0, 630.0, 200.0, step=5.0)
    membership = st.selectbox("Membership", ["Yes", "No"], index=0)

user_input = {
    "age": age,
    "gender": gender,
    "income": income,
    "region": region,
    "avg_spend": avg_spend,
    "membership": membership,
}

st.divider()

# ── Predict ───────────────────────────────────────────────────────────────────
if st.button("🎯  Predict Satisfaction Score", use_container_width=True):
    input_df = prepare_input_for_prediction(user_input)
    result   = predict_satisfaction(model_bundle, input_df)
    rec      = get_satisfaction_recommendation(result)

    SAT_COLORS = {0: "#EF4444", 1: "#F59E0B", 2: "#10B981"}
    color = SAT_COLORS[result["class"]]

    r1, r2, r3 = st.columns(3, gap="small")
    with r1:
        st.markdown(f"""
        <div class="result-card" style="background:{color}18;border:1px solid {color}44;">
            <div class="rc-label">Predicted Satisfaction Score</div>
            <div class="rc-value" style="color:{color};">{result['predicted_score']:.2f} <span style="font-size:1rem;color:#94A3B8;">/ 5.0</span></div>
        </div>""", unsafe_allow_html=True)
    with r2:
        st.markdown(f"""
        <div class="result-card" style="background:rgba(99,102,241,0.1);border:1px solid rgba(99,102,241,0.25);">
            <div class="rc-label">Satisfaction Tier</div>
            <div class="rc-value" style="color:#A5B4FC;">{result['prediction']}</div>
        </div>""", unsafe_allow_html=True)
    with r3:
        st.markdown(f"""
        <div class="result-card" style="background:rgba(16,185,129,0.08);border:1px solid rgba(16,185,129,0.25);">
            <div class="rc-label">Prediction Confidence</div>
            <div class="rc-value" style="color:#34D399;">{result['confidence']:.1f}%</div>
        </div>""", unsafe_allow_html=True)

    st.divider()

    # Probability bars
    st.markdown("""
    <div class="section-header">
        <div class="section-accent"></div>
        <h2>Class Probabilities</h2>
    </div>""", unsafe_allow_html=True)

    labels = ["Low Satisfaction (≤2.0)", "Moderate Satisfaction (2.1–3.5)", "High Satisfaction (>3.5)"]
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
            <i class="fa-solid fa-lightbulb" style="color:{rec_color};margin-right:0.4rem;"></i> Strategic Recommendation
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

fi_df = get_feature_importance(model_bundle)
fig_fi = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                color="Importance", color_continuous_scale=["#1F2D45","#6366F1","#A5B4FC"],
                text=fi_df["Importance"].apply(lambda v: f"{v:.3f}"))
fig_fi.update_traces(textposition="outside")
fig_fi.update_layout(**get_chart_layout(height=340, show_legend=False, title="RandomForest Feature Importance"))
fig_fi.update_coloraxes(showscale=False)
st.plotly_chart(fig_fi, use_container_width=True)
