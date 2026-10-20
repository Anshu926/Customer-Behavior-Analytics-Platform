"""
Satisfaction Predictor — 10k Consumer Behaviour Dataset
Uses 6 input features: Age, Gender, Income_Level, Region, Avg_Spend, Loyalty Member.
Predicts Satisfaction_Class (Low, Moderate, High) with thresholds: <=2.0 = Low, <=3.5 = Moderate, >3.5 = High.
Excludes Customer_ID, Satisfaction_Score, Purchase_Category, and Payment_Mode from inputs.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from utils import (
    load_data, load_satisfaction_model, load_score_regressor, render_sidebar,
    get_chart_layout, prepare_input_for_prediction, predict_satisfaction,
    get_satisfaction_recommendation, get_feature_importance, FA_CDN,
    GENDER_MAPPING, INCOME_MAPPING, REGION_MAPPING,
    LOYALTY_MAPPING, SATISFACTION_MAPPING
)

st.set_page_config(page_title="Satisfaction Predictor", page_icon="🎯", layout="wide", initial_sidebar_state="expanded")
render_sidebar("prediction")

@st.cache_resource
def get_models():
    df = load_data()
    clf = load_satisfaction_model(df)
    reg = load_score_regressor(df)
    return clf, reg, df

model, reg_model, df = get_models()

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown(f"""
{FA_CDN}
<div class="hero-banner" style="padding:1.6rem 2rem;">
    <div class="hero-tag"><i class="fa-solid fa-bullseye"></i> ML Prediction</div>
    <h1>Satisfaction Predictor</h1>
    <p class="hero-sub">Forecast customer satisfaction class and numerical score using Random Forest models trained on 10,000 consumer behaviour records with stratified validation and no target leakage.</p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── Input form (Strictly 6 Features) ──────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="section-accent"></div>
    <h2>Customer Profile Input (6 Core Features)</h2>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="medium")

with col1:
    age = st.slider("Customer Age", 18, 69, 35, help="Age of the customer in years (18–69)")
    gender = st.selectbox(
        "Gender",
        ["F", "M", "Other"],
        format_func=lambda x: {"F": "Female (F)", "M": "Male (M)", "Other": "Other"}.get(x, x),
        help="Customer gender"
    )
    income = st.selectbox("Income Level", ["Low", "Medium", "High"], index=1, help="Household income bracket")

with col2:
    avg_spend = st.slider("Average Spend ($)", 10.0, 630.0, 200.0, step=5.0, help="Average spend amount per purchase in USD")
    region = st.selectbox("Region", ["Rural", "Semi-Urban", "Urban"], index=2, help="Geographical residential region")
    loyalty = st.selectbox("Loyalty Member", ["No", "Yes"], index=1, help="Enrolled in customer loyalty programme")

user_input = {
    "age": age,
    "gender": gender,
    "income": income,
    "region": region,
    "avg_spend": avg_spend,
    "loyalty": loyalty,
}

st.divider()

# ── Predict ───────────────────────────────────────────────────────────────────
if st.button("🎯  Predict Satisfaction", use_container_width=True):
    input_df = prepare_input_for_prediction(user_input)
    result = predict_satisfaction(model, input_df, reg_model=reg_model)
    rec = get_satisfaction_recommendation(result)

    SAT_COLORS = {0: "#EF4444", 1: "#F59E0B", 2: "#10B981"}
    color = SAT_COLORS[result["class"]]
    tier_ranges = {0: "Score ≤ 2.0", 1: "2.1 ≤ Score ≤ 3.5", 2: "Score > 3.5"}

    # ── 3 Result Cards ────────────────────────────────────────────────────────
    r1, r2, r3 = st.columns(3, gap="small")
    with r1:
        st.markdown(f"""
        <div class="result-card" style="background:{color}18;border:1px solid {color}44;">
            <div class="rc-label">Predicted Outcome</div>
            <div class="rc-value" style="color:{color};">{result['prediction']}</div>
            <div style="font-size:0.75rem;color:#94A3B8;margin-top:0.35rem;">
                Threshold: <strong>{tier_ranges[result['class']]}</strong>
            </div>
        </div>""", unsafe_allow_html=True)

    with r2:
        if result.get("pred_score") is not None:
            score_display = f"{result['pred_score']:.2f} / 5.0"
            st.markdown(f"""
            <div class="result-card" style="background:rgba(16,185,129,0.08);border:1px solid rgba(16,185,129,0.25);">
                <div class="rc-label">Predicted Satisfaction Score</div>
                <div class="rc-value" style="color:#34D399;">{score_display}</div>
                <div style="font-size:0.75rem;color:#94A3B8;margin-top:0.35rem;">
                    Random Forest Regressor Estimate
                </div>
            </div>""", unsafe_allow_html=True)
        else:
            class_name = ["Low", "Moderate", "High"][result['class']]
            st.markdown(f"""
            <div class="result-card" style="background:rgba(16,185,129,0.08);border:1px solid rgba(16,185,129,0.25);">
                <div class="rc-label">Predicted Satisfaction Score</div>
                <div class="rc-value" style="color:#34D399;">{class_name}</div>
                <div style="font-size:0.75rem;color:#94A3B8;margin-top:0.35rem;">
                    Classification Tier
                </div>
            </div>""", unsafe_allow_html=True)

    with r3:
        st.markdown(f"""
        <div class="result-card" style="background:rgba(99,102,241,0.1);border:1px solid rgba(99,102,241,0.25);">
            <div class="rc-label">Model Confidence</div>
            <div class="rc-value" style="color:#A5B4FC;">{result['confidence']:.1f}%</div>
            <div style="font-size:0.75rem;color:#94A3B8;margin-top:0.35rem;">
                Predicted-Class Probability
            </div>
        </div>""", unsafe_allow_html=True)

    st.divider()

    # ── Probability Distribution (All 3 Classes) ──────────────────────────────
    st.markdown("""
    <div class="section-header">
        <div class="section-accent"></div>
        <h2>Class Probability Distribution (3 Classes)</h2>
    </div>""", unsafe_allow_html=True)

    class_names = ["Low Satisfaction (≤ 2.0)", "Moderate Satisfaction (2.1–3.5)", "High Satisfaction (> 3.5)"]
    class_colors = ["#EF4444", "#F59E0B", "#10B981"]

    prob_df = pd.DataFrame({
        "Class": class_names,
        "Probability": [p * 100 for p in result["probas"]],
        "Color": class_colors,
        "Formatted": [f"{p * 100:.1f}%" for p in result["probas"]]
    })

    fig_prob = go.Figure()
    for row in prob_df.itertuples():
        fig_prob.add_trace(go.Bar(
            name=row.Class,
            y=[row.Class],
            x=[row.Probability],
            orientation="h",
            marker=dict(color=row.Color, line=dict(width=0)),
            text=[row.Formatted],
            textposition="outside",
            textfont=dict(size=12, color="#F1F5F9")
        ))

    lay_prob = get_chart_layout(height=240, show_legend=False, title="")
    lay_prob["xaxis"] = dict(title="Probability (%)", range=[0, 105], gridcolor="#1F2D45")
    lay_prob["yaxis"] = dict(autorange="reversed")
    fig_prob.update_layout(**lay_prob)
    st.plotly_chart(fig_prob, use_container_width=True)

    # ── Actionable Recommendation ─────────────────────────────────────────────
    rec_color = SAT_COLORS[result["class"]]
    st.markdown(f"""
    <div style="background:rgba(17,24,39,0.9);border:1px solid {rec_color}44;border-left:4px solid {rec_color};
                border-radius:10px;padding:1.1rem 1.4rem;margin-top:0.5rem;">
        <div style="color:#94A3B8;font-size:0.7rem;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:0.45rem;">
            <i class="fa-solid fa-lightbulb" style="color:{rec_color};margin-right:0.4rem;"></i> Actionable Recommendation
        </div>
        <div style="color:#F1F5F9;font-size:0.92rem;line-height:1.6;">{rec}</div>
    </div>""", unsafe_allow_html=True)

    st.divider()

# ── Feature Importance Visualization ──────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="section-accent"></div>
    <h2>Feature Importance (6 Core Predictors)</h2>
</div>""", unsafe_allow_html=True)

fi_df = get_feature_importance(model)
fig_fi = px.bar(
    fi_df,
    x="Importance",
    y="Feature",
    orientation="h",
    color="Importance",
    color_continuous_scale=["#1F2D45", "#6366F1", "#A5B4FC"],
    text=fi_df["Importance"].apply(lambda v: f"{v:.3f}"),
    labels={"Importance": "Relative Feature Importance", "Feature": "Input Feature"}
)
fig_fi.update_traces(textposition="outside", textfont=dict(size=11, color="#F1F5F9"), width=0.55)
lay_fi = get_chart_layout(height=340, show_legend=False, title="")
lay_fi["xaxis"] = dict(title="Importance Weight", gridcolor="#1F2D45")
lay_fi["yaxis"] = dict(title="")
fig_fi.update_layout(**lay_fi)
fig_fi.update_coloraxes(showscale=False)
st.plotly_chart(fig_fi, use_container_width=True)

st.markdown("""
<div style="text-align:center;padding:1rem 0;color:#2D3B55;font-size:0.75rem;">
    Satisfaction Predictor • 6 Features: Age, Gender, Income Level, Region, Average Spend, Loyalty Member
</div>
""", unsafe_allow_html=True)
