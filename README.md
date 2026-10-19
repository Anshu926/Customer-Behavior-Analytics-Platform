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
│   ├── prediction.py          # Satisfaction predictor (RandomForest)
│   ├── segmentation.py        # Customer segmentation (KMeans)
│   └── insights.py            # Strategic business insights
│
├── customer_data.csv          # Raw 10,000-row dataset
├── clean_customer_data.csv    # Preprocessed & encoded dataset (for ML)
│
├── predictor.pkl              # Trained RandomForestClassifier
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
| **Purpose** | Consumer behaviour analysis, ML model training |

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

| Column | Encoding |
|---|---|
| `Gender_Enc` | F=0, M=1, Other=2 |
| `Income_Enc` | Low=0, Medium=1, High=2 |
| `Region_Enc` | Rural=0, Semi-Urban=1, Urban=2 |
| `Category_Enc` | Apparel=0, Electronics=1, Groceries=2, Health & Beauty=3, Home & Kitchen=4, Sports & Fitness=5 |
| `Payment_Enc` | Cash=0, Credit Card=1, Debit Card=2, UPI=3, Wallet=4 |
| `Loyalty_Enc` | No=0, Yes=1 |
| `Satisfaction_Class` | Score ≤2.0 → 0 (Low), ≤3.5 → 1 (Moderate), >3.5 → 2 (High) |

---

## 🚀 Platform Modules

### 1. 🏠 Home / Landing Page (`app.py`)
- Hero banner with platform overview
- Live KPI ribbon: **Total Customers, Average Spend, Total Revenue, Avg Satisfaction Score**
- Platform module cards
- Dataset snapshot: loyalty breakdown, satisfaction tiers, age stats
- Technology stack overview

---

### 2. 📊 Business Dashboard (`pages/dashboard.py`)
- **5 KPI cards** — Customers, Avg Spend, Total Revenue, Avg Score, Loyalty Rate
- **Satisfaction distribution** — pie chart of Low / Moderate / High classes
- **Category analysis** — avg spend & score by purchase category
- **Regional performance** — spend & satisfaction by region
- **Income level breakdown** — bar charts across income tiers
- **Payment mode analysis** — distribution of payment preferences
- **Gender & age demographics** — gender split, age histogram
- **Spend vs. Satisfaction scatter** — coloured by loyalty membership

---

### 3. 🎯 Satisfaction Predictor (`pages/prediction.py`)
- **ML model:** RandomForestClassifier trained on 10,000 records
- **Input features:**
  - Age, Gender, Income Level, Region
  - Purchase Category, Avg Spend, Payment Mode, Loyalty Member
- **Output:**
  - Predicted satisfaction class (Low / Moderate / High)
  - Probability bar chart for all three classes
  - Personalised business recommendation based on prediction
  - Top-5 feature importance chart

**Model Training Details:**

| Property | Value |
|---|---|
| Algorithm | RandomForestClassifier |
| Training samples | 10,000 |
| Target | Satisfaction_Class (3 classes) |
| Features | 8 encoded features |
| Persistence | `predictor.pkl` (joblib) |

---

### 4. 🔵 Customer Segmentation (`pages/segmentation.py`)
- **ML model:** KMeans clustering (k=4) with StandardScaler normalisation
- **4 Customer Personas:**

| Cluster | Label | Description |
|---|---|---|
| 0 | 🏆 High-Value Loyalists | High spend, loyalty members, top satisfaction |
| 1 | 💰 Budget Shoppers | Lower spend, price-sensitive, mixed satisfaction |
| 2 | 💎 Premium Explorers | High income, varied categories, non-loyalty |
| 3 | 🛒 Occasional Buyers | Infrequent purchases, low–moderate engagement |

- Cluster summary cards with avg spend, score, loyalty rate
- Scatter plots: Spend vs Satisfaction, Age vs Spend (coloured by cluster)
- Category & income breakdown per cluster
- **Classify-a-new-customer** form → instant cluster assignment

**Clustering Details:**

| Property | Value |
|---|---|
| Algorithm | KMeans (k=4) |
| Features | Age, Income_Enc, Avg_Spend, Category_Enc, Loyalty_Enc |
| Normalisation | StandardScaler |
| Persistence | `cluster.pkl` + `scaler.pkl` |

---

### 5. 💡 Business Insights (`pages/insights.py`)
Five automated data-driven insights:

1. **Top Purchase Category** — highest avg spend & satisfaction driver
2. **Loyalty Impact** — satisfaction score delta: members vs. non-members
3. **Regional Performance** — best & worst performing regions
4. **Payment Mode vs. Satisfaction** — which payment method correlates with happiest customers
5. **Age Group Patterns** — satisfaction trends across age cohorts

Plus a **Strategic Recommendations** section with 4 action cards (Loyalty Programme, Category Marketing, Regional Strategy, Payment Optimisation).

---

## 🛠️ Technology Stack

| Layer | Technologies |
|---|---|
| **Frontend** | Streamlit 1.53+, Plotly 6.0, Custom CSS Design System |
| **Backend** | Python 3.9+, Pandas 2.2, NumPy 2.1 |
| **Machine Learning** | scikit-learn 1.5 (RandomForest, KMeans, StandardScaler) |
| **Persistence** | Joblib 1.5 |
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

### 3. Run the app
```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

> **Note:** ML models (`predictor.pkl`, `cluster.pkl`, `scaler.pkl`) are included in the repository. If they are missing, the app will auto-train them on first launch using `clean_customer_data.csv`.

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
| Avg Satisfaction Score | ~3.25 / 5.0 |
| Loyalty Members | ~50% |
| Purchase Categories | 6 |
| Customer Segments | 4 |
| Satisfaction Classes | 3 (Low / Moderate / High) |
| ML Models | 2 (RandomForest + KMeans) |

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

- Dataset generated synthetically for research and educational use
- Built with [Streamlit](https://streamlit.io), [scikit-learn](https://scikit-learn.org), and [Plotly](https://plotly.com)
