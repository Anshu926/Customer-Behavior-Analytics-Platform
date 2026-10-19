# 📊 Customer Behavior Analytics Platform

> An enterprise-grade AI-powered customer intelligence platform built with **Streamlit**, **scikit-learn**, **Pandas**, and **Plotly** — powered by a **10,000-record consumer behaviour dataset**.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://customer-behavior-analytics-platform-app.streamlit.app)

---

## 🌐 Live Demo

**[→ Open the App](https://customer-behavior-analytics-platform-app.streamlit.app)**

---

## 📁 Project Structure

```
Customer Behavior Analytics Platform/
│
├── app.py                     # Landing page / home
├── utils.py                   # Shared utilities, ML loaders, CSS design system
├── requirements.txt           # Pinned dependencies
│
├── pages/
│   ├── dashboard.py           # Business dashboard & KPI visualisations
│   ├── prediction.py          # Satisfaction predictor (Satisfaction_Score & Tier)
│   ├── segmentation.py        # Customer segmentation (KMeans)
│   └── insights.py            # Strategic business insights
│
├── customer_data.csv          # 10,000-row dataset (7 core features)
├── clean_customer_data.csv    # Preprocessed & encoded dataset (for ML)
│
├── predictor.pkl              # Trained RandomForest models (Classifier + Regressor)
├── cluster.pkl                # Trained KMeans (4 clusters)
└── scaler.pkl                 # StandardScaler for clustering
```

---

## 📦 Dataset & Schema

This platform uses 7 core customer attributes designed for behavioral intelligence and satisfaction prediction:

| Feature | Type | Description | Values / Range |
|---|---|---|---|
| `Gender` | String | Customer gender | `F`, `M`, `Other` |
| `Age` | Integer | Customer age in years | `18` – `69` |
| `Income_Level` | String | Income bracket | `Low`, `Medium`, `High` |
| `Region` | String | Geographic market | `Rural`, `Semi-Urban`, `Urban` |
| `Avg_Spend` | Float | Average transaction spend ($) | `$10.00` – `$629.89` |
| `Membership` | String | Membership enrollment status | `Yes`, `No` |
| `Satisfaction_Score` | Float | Target satisfaction rating | `1.0` – `5.0` |

### Engineered Features (in `clean_customer_data.csv`)

| Column | Encoding / Mapping |
|---|---|
| `Gender_Enc` | `F` = 0, `M` = 1, `Other` = 2 |
| `Income_Enc` | `Low` = 0, `Medium` = 1, `High` = 2 |
| `Region_Enc` | `Rural` = 0, `Semi-Urban` = 1, `Urban` = 2 |
| `Membership_Enc` | `No` = 0, `Yes` = 1 |
| `Satisfaction_Class` | `Score ≤ 2.0` → 0 (Low), `2.1 – 3.5` → 1 (Moderate), `> 3.5` → 2 (High) |

---

## 🚀 Platform Modules

### 1. 🏠 Home / Landing Page (`app.py`)
- Live executive metrics ribbon: **Total Customers, Average Spend, Total Revenue, Avg Satisfaction Score**
- Platform module navigation and feature summaries
- Dataset snapshot: membership rate, satisfaction tier distribution, age statistics

### 2. 📊 Business Dashboard (`pages/dashboard.py`)
- **5 KPI cards** — Total Customers, Avg Spend, Avg Satisfaction, Total Revenue, Membership Rate
- **Satisfaction score distribution** — Pie chart (Tiers) and Histogram (1.0 to 5.0)
- **Membership impact analysis** — Spend and Satisfaction comparisons (Members vs Non-Members)
- **Region & Income analysis** — Spend by Region and Satisfaction by Income Level
- **Demographics overview** — Gender breakdown and Age distribution by Gender
- **Spend vs Satisfaction scatter plot** — Colored by Membership with regression trendline

### 3. 🎯 Satisfaction Predictor (`pages/prediction.py`)
- **Predicts:**
  - **Satisfaction_Score (1.0 – 5.0)** using `RandomForestRegressor`
  - **Satisfaction Tier (Low / Moderate / High)** using `RandomForestClassifier`
  - **Model Confidence Score (%)**
- **Inputs:** `Gender`, `Age`, `Income_Level`, `Region`, `Avg_Spend`, `Membership`
- **Visuals:** Class probability distribution bars and feature importance rankings
- **Actionable Advice:** Tailored strategic recommendations based on prediction

### 4. 🔵 Customer Segmentation (`pages/segmentation.py`)
- **ML Algorithm:** KMeans (k=4) with `StandardScaler`
- **Segments:**
  - 🏆 **Male Loyalty Members** — High loyalty enrollment, steady spend
  - 🛍️ **Female Non-Members** — High conversion potential for membership programs
  - 💎 **Female Loyalty Members** — Engaged brand champions with high satisfaction
  - 🛒 **Male Non-Members** — Independent buyers with distinct purchase patterns
- **Interactive features:** Scatter plots (Spend vs Satisfaction, Age vs Spend) and instant customer profile classification tool

### 5. 💡 Business Insights (`pages/insights.py`)
- **5 Strategic Insights:**
  1. **Membership Program Value** — Quantified spend & satisfaction lift for members
  2. **Income Tier Dynamics** — Spending behavior across High, Medium, and Low income
  3. **Regional Spending Leaders** — Regional market penetration and spend volume
  4. **Gender Demographics & Satisfaction** — Cross-gender engagement benchmarks
  5. **Age Cohort Behaviors** — Spending trends across generation brackets
- **Strategic Recommendations:** Actionable playbooks for membership growth, high-income targeting, and retention

---

## 🛠️ Technology Stack

| Layer | Technologies |
|---|---|
| **Frontend** | Streamlit 1.53+, Plotly 6.0, Custom Dark Theme CSS |
| **Backend** | Python 3.9+, Pandas 2.2, NumPy 2.1 |
| **Machine Learning** | scikit-learn 1.5 (RandomForest Classifier & Regressor, KMeans, StandardScaler) |
| **Persistence** | Joblib 1.5 |
| **Deployment** | Streamlit Community Cloud |

---

## ⚙️ Running Locally

```bash
# 1. Clone repository
git clone https://github.com/Anshu926/Customer-Behavior-Analytics-Platform.git
cd Customer-Behavior-Analytics-Platform

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the application
streamlit run app.py
```

Open **http://localhost:8501** in your browser.
