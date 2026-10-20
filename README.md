# 📊 Customer Behavior Analytics Platform

> An enterprise-grade AI-powered customer intelligence platform built with **Streamlit**, **scikit-learn**, **Pandas**, and **Plotly** — powered by a **10,000-record synthetic consumer behaviour dataset**.

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
│   ├── prediction.py          # Satisfaction predictor (Dual-model Random Forest)
│   ├── segmentation.py        # Customer segmentation & cluster comparison
│   └── insights.py            # Strategic business insights
│
├── customer_data.csv          # Raw 10,000-row dataset
├── clean_customer_data.csv    # Preprocessed & encoded dataset (for ML)
│
├── predictor.pkl              # Trained RandomForestClassifier (Satisfaction_Class)
├── regressor.pkl              # Trained RandomForestRegressor (Satisfaction_Score)
├── cluster.pkl                # Trained KMeans (4 clusters)
└── scaler.pkl                 # StandardScaler for clustering
```

---

## 📦 Dataset

This project is built on a **synthetic consumer behaviour dataset** generated for research and educational purposes.

| Property | Value |
|---|---|
| **Records** | 10,000 unique customers |
| **Source** | Synthetic (research-grade) |
| **Purpose** | Consumer behaviour analysis, ML classification & clustering |

### Schema

| Column | Type | Description | Sample Values |
|---|---|---|---|
| `Customer_ID` | String | Unique customer identifier | C00001 – C10000 |
| `Age` | Integer | Customer age in years | 18 – 69 |
| `Gender` | String | Customer gender | F, M, Other |
| `Income_Level` | String | Income bracket | Low, Medium, High |
| `Region` | String | Geographic location | Rural, Semi-Urban, Urban |
| `Purchase_Category` | String | Primary product category | Apparel, Electronics, Groceries, Health & Beauty, Home & Kitchen, Sports & Fitness |
| `Avg_Spend` | Float | Average spend per transaction (\$) | 10.00 – 629.89 |
| `Payment_Mode` | String | Preferred payment method | Cash, Credit Card, Debit Card, UPI, Wallet |
| `Loyalty_Member` | String | Loyalty programme membership | Yes, No |
| `Satisfaction_Score` | Float | Customer satisfaction rating | 1.0 – 5.0 |

### Engineered Features (in `clean_customer_data.csv`)

| Column | Encoding | Description |
|---|---|---|
| `Gender_Enc` | F=0, M=1, Other=2 | Categorical numeric mapping |
| `Income_Enc` | Low=0, Medium=1, High=2 | Ordinal income tier mapping |
| `Region_Enc` | Rural=0, Semi-Urban=1, Urban=2 | Location category encoding |
| `Category_Enc` | Apparel=0, Electronics=1, Groceries=2, Health & Beauty=3, Home & Kitchen=4, Sports & Fitness=5 | Product category encoding |
| `Payment_Enc` | Cash=0, Credit Card=1, Debit Card=2, UPI=3, Wallet=4 | Transaction mode encoding |
| `Loyalty_Enc` | No=0, Yes=1 | Binary membership indicator |
| `Satisfaction_Class` | Score ≤2.0 → 0 (Low), ≤3.5 → 1 (Moderate), >3.5 → 2 (High) | Target discrete satisfaction tier |

---

## 🚀 Platform Modules

### 1. 🏠 Home / Landing Page (`app.py`)
- Hero banner with platform overview and real-time intelligence summary.
- Live KPI ribbon: **Total Customers, Average Spend, Total Revenue, Avg Satisfaction Score**.
- Platform module directory and navigation cards.
- Dataset snapshot: loyalty breakdown, satisfaction tier distribution, and age demographics.
- Technology stack overview and production-readiness badge indicators.

---

### 2. 📊 Business Dashboard (`pages/dashboard.py`)
- **5 KPI cards** — Total Customers, Avg Spend, Total Revenue, Avg Score, Loyalty Rate.
- **Satisfaction distribution** — Donut chart of Low / Moderate / High classes alongside a score histogram.
- **Category analysis** — Customer distribution and avg spend by product category.
- **Regional performance** — Average spend comparison across Rural, Semi-Urban, and Urban segments.
- **Income level breakdown** — Satisfaction levels across Low, Medium, and High income brackets.
- **Payment mode analysis** — Popularity and revenue share by payment gateway.
- **Gender & age demographics** — Bimodal age histogram overlaid by gender.
- **Spend vs. Satisfaction scatter** — Correlation plot with trendline colored by income tier.

---

### 3. 🎯 Satisfaction Predictor (`pages/prediction.py`)
A streamlined, high-accuracy inference engine using **strictly 6 core features**:
- **Input Features:** `Age`, `Gender`, `Income_Level`, `Region`, `Avg_Spend`, `Loyalty_Member`.
- **Excluded Features:** `Customer_ID`, `Satisfaction_Score`, `Purchase_Category`, `Payment_Mode` (no target leakage).
- **Dual-Model Architecture:**
  1. **Random Forest Classifier**: Trained with a stratified 80/20 train/test split on `Satisfaction_Class`.
  2. **Random Forest Regressor**: Trained to estimate continuous `Satisfaction_Score` (1.0–5.0).
- **3 Dynamic Result Cards:**
  1. **Predicted Outcome**: Classification tier (*Low*, *Moderate*, *High*) with satisfaction threshold indicator.
  2. **Predicted Satisfaction Score**: Regression estimate (*e.g., 3.30 / 5.0*).
  3. **Model Confidence**: Probability of the predicted class (*e.g., 45.0%*).
- **Probability Distribution**: Horizontal bar visualization showing confidence across all 3 classes.
- **Feature Importance**: Ranked visualization of the 6 predictor weights.

---

### 4. 🔵 Customer Segmentation (`pages/segmentation.py`)
- **Unsupervised ML:** KMeans clustering ($k=4$) with `StandardScaler` normalization over 5 dimensions (`Age`, `Income_Enc`, `Avg_Spend`, `Category_Enc`, `Loyalty_Enc`).
- **4 Data-Derived Customer Personas:**

| Cluster | Assigned Persona | Actual Characteristics & Profile | Distribution |
| :---: | :--- | :--- | :---: |
| **C0** | **Young Loyalists** | Younger cohort (mean age ~31.0 yrs), **100% Loyalty Members**, Low/Medium income | 2,448 (24.5%) |
| **C1** | **Mature Loyalists** | Senior demographic (mean age ~56.9 yrs), **100% Loyalty Members**, highest spend ($204) | 2,358 (23.6%) |
| **C2** | **High-Income Affluents** | **100% High Income tier**, age ~43.7 yrs, moderate loyalty enrollment (58.7%) | 2,003 (20.0%) |
| **C3** | **Standard Non-Members** | General demographic (~43.2 yrs), **0.0% Loyalty Members** (unenrolled) | 3,191 (31.9%) |

- **Segment Overview Cards**: Dynamic metrics for customer count, spend, score, and loyalty percentage per cluster.
- **Cluster Visualisation**: Scatter plots (`Spend vs Satisfaction` and `Age vs Spend`) with consistent color mapping.
- **Segment Behaviour Breakdown**: Multi-dimensional category preference and income mix charts.
- **Cluster Comparison Dashboard**:
  - Customer Count & Percentage Share
  - Average Spend Comparison ($)
  - Average Satisfaction Rating
  - Membership Rate (%)
  - Summary Data Table

---

### 5. 💡 Business Insights (`pages/insights.py`)
Five automated, data-driven analytical modules:
1. **Top Purchase Category** — Revenue and satisfaction leader.
2. **Loyalty Impact** — Monetary and retention lift comparing members vs. non-members.
3. **Regional Performance** — Top spending geographies and customer volume.
4. **Payment Mode vs. Satisfaction** — Customer happiness correlated with payment methods.
5. **Age Group Dynamics** — Behavioral shifts across age brackets.

Includes an **Actionable Recommendations** matrix tailored for marketing, retention, and growth operations.

---

## 🛠️ Technology Stack

| Layer | Technologies |
|---|---|
| **Frontend** | Streamlit 1.53+, Plotly 6.0, Custom CSS Dark Design System |
| **Backend** | Python 3.9+, Pandas 2.2, NumPy 2.1 |
| **Machine Learning** | scikit-learn 1.5 (RandomForestClassifier, RandomForestRegressor, KMeans, StandardScaler) |
| **Model Persistence** | Joblib 1.5 |
| **Deployment** | Streamlit Community Cloud |

---

## ⚙️ Running Locally

### 1. Clone the repository
```bash
git clone https://github.com/Anshu926/Customer-Behavior-Analytics-Platform.git
cd Customer-Behavior-Analytics-Platform
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the application
```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

> **Note:** Serialized ML models (`predictor.pkl`, `regressor.pkl`, `cluster.pkl`, `scaler.pkl`) are pre-trained and included in the repository. If absent, the application automatically retrains them on first launch from `clean_customer_data.csv`.

---

## 📋 Requirements

```
streamlit==1.53.1
pandas==2.2.3
numpy==2.1.3
scikit-learn==1.5.2
plotly==6.0.0
joblib==1.5.3
```

---

## 📈 Key Statistics

| Metric | Value |
|---|---|
| Total Customers | 10,000 |
| Avg Satisfaction Score | ~3.00 / 5.0 |
| Loyalty Program Members | 59.8% |
| Product Categories | 6 |
| Customer Segments | 4 |
| Satisfaction Classes | 3 (Low / Moderate / High) |
| Trained ML Models | 3 (Random Forest Classifier, Random Forest Regressor, KMeans) |

---

## 🗺️ App Navigation

```
📊 Home (app.py)
│
├── 📊 Dashboard     → pages/dashboard.py
├── 🎯 Predictor     → pages/prediction.py
├── 🔵 Segmentation  → pages/segmentation.py
└── 💡 Insights      → pages/insights.py
```

---

## 🙏 Acknowledgements

- Dataset generated synthetically for research and educational purposes.
- Built with [Streamlit](https://streamlit.io), [scikit-learn](https://scikit-learn.org), and [Plotly](https://plotly.com).
