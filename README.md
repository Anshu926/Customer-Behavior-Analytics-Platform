# 📊 Customer Behavior Analytics Platform

> An enterprise-grade AI-powered customer intelligence platform built with **Streamlit**, **scikit-learn**, **Pandas**, and **Plotly** — powered by a **10,000-record synthetic consumer behaviour dataset** across 14 detailed attributes.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://customer-behavior-analytics-platform-app.streamlit.app)

---

## 🌐 Live Demo & Deployment

- **Live URL:** [https://customer-behavior-analytics-platform-app.streamlit.app](https://customer-behavior-analytics-platform-app.streamlit.app)
- **Local URL:** `http://localhost:8501`

---

## 📁 Project Structure

```
Customer Behavior Analytics Platform/
│
├── app.py                     # Landing page, data source attribution & dataset downloads
├── utils.py                   # Data loaders, ML pipeline, design system & styling
├── requirements.txt           # Pinned dependencies
│
├── pages/
│   ├── dashboard.py           # Business dashboard, KPIs & behavioral charts
│   ├── prediction.py          # Satisfaction predictor (Dual-model RF: Tier & 1-5 Score)
│   ├── segmentation.py        # Customer segmentation (KMeans - 4 personas & insights)
│   └── insights.py            # Automated diagnostic business insights
│
├── customer_data.csv          # Raw 10,000-row dataset (14 columns)
├── clean_customer_data.csv    # Preprocessed & encoded dataset (24 columns)
│
├── predictor.pkl              # Trained RandomForestClassifier (Satisfaction_Class)
├── regressor.pkl              # Trained RandomForestRegressor (Satisfaction_Score 1-5)
├── cluster.pkl                # Trained KMeans (4 clusters)
└── scaler.pkl                 # StandardScaler for clustering
```

---

## 📦 Dataset Schema (10,000 Records × 14 Columns)

| Column | Type | Description | Sample Values |
|---|---|---|---|
| `Customer_ID` | String | Unique customer identifier | CUST00001 – CUST10000 |
| `Age` | Integer | Customer age in years | 18 – 69 |
| `Gender` | String | Customer gender | F, M, Other |
| `Income_Level` | String | Income bracket | Low, Medium, High |
| `Region` | String | Geographic location | Rural, Semi-Urban, Urban |
| `Purchase_Category` | String | Primary product category | Apparel, Electronics, Groceries, Health & Beauty, Home & Kitchen, Sports & Fitness |
| `Avg_Spend` | Float | Average spend per transaction (\$) | 10.00 – 629.89 |
| `Payment_Mode` | String | Preferred payment method | Cash, Credit Card, Debit Card, UPI, Wallet |
| `Membership_Status` | String | Loyalty / membership program enrollment | Yes, No |
| `Items_Purchased` | Integer | Basket size / item count | 1 – 20 items |
| `Discount_Applied` | Boolean | Whether discount was used | True, False |
| `Days_Since_Last_Purchase`| Integer | Purchase recency interval | 1 – 90 days |
| `Satisfaction_Score` | Float | Numerical review score | 1.0 – 5.0 |
| `Satisfaction_Level` | String | Customer satisfaction classification | Low (<=2.0), Moderate (2.1-3.5), High (>=3.6) |

### Feature Engineering (`clean_customer_data.csv`)

| Engineered Feature | Mapping |
|---|---|
| `Gender_Enc` | F: 0, M: 1, Other: 2 |
| `Income_Enc` | Low: 0, Medium: 1, High: 2 |
| `Region_Enc` | Rural: 0, Semi-Urban: 1, Urban: 2 |
| `Category_Enc` | Apparel: 0, Electronics: 1, Groceries: 2, Health & Beauty: 3, Home & Kitchen: 4, Sports & Fitness: 5 |
| `Payment_Enc` | Cash: 0, Credit Card: 1, Debit Card: 2, UPI: 3, Wallet: 4 |
| `Membership_Enc` | No: 0, Yes: 1 |
| `Discount_Enc` | False: 0, True: 1 |
| `Satisfaction_Class` | Low: 0 (590), Moderate: 1 (5,755), High: 2 (3,655) |

---

## 📦 Data Sources & Dataset Downloads

The platform is driven by 10,000 verified multi-dimensional consumer behaviour records sourced and cross-referenced from benchmark open data repositories:

1. **Kaggle Source:** [E-Commerce Customer Behavior Dataset](https://www.kaggle.com/datasets/uom190346a/e-commerce-customer-behavior-dataset) by `@uom190346a`
2. **Zenodo Open Access:** [Artificial Consumer Shopping Behavior Data for Marketing and Analytics](https://zenodo.org/records/17036147) by Manik Sonawane (DOI: `10.5281/zenodo.17036147`)

Both datasets are accessible and directly downloadable from the home page:
- **`customer_data.csv` (Raw Dataset):** 10,000 rows × 14 columns — Original text categories and ground-truth records.
- **`clean_customer_data.csv` (Processed Dataset):** 10,000 rows × 24 columns — Cleaned, transformed, and feature-engineered with 10 numeric ML encodings.

---

## 🚀 Platform Modules

### 1. 🏠 Home Page (`app.py`)
- Hero banner with enterprise dark aesthetic
- Live KPI ribbon: **Total Customers, Average Spend, Total Revenue, Avg Satisfaction**
- **Dataset Information & Attribution:** Dual source links (Kaggle & Zenodo) and direct CSV download options
- Modernized 4-card **Technology Stack** and interactive project footer banner

### 2. 📊 Business Dashboard (`pages/dashboard.py`)
- **6 Executive KPI Cards:** Total Customers, Avg Spend, Total Revenue, Avg Satisfaction, Avg Basket Size, Membership Rate
- **Satisfaction Analysis:** Donut chart & continuous score distribution histogram
- **Category Dynamics:** Customer volume & average spend by category
- **Shopping Behavioral Dynamics:** Basket size vs spend, discount elasticity analysis, recency cohorts
- **Geographic & Economic Breakdown:** Regional performance and income level satisfaction
- **Demographics & Payment Methods:** Payment preferences & age/gender distributions
- **Correlation Engine:** Spend vs satisfaction scatter with trendline

### 3. 🎯 Satisfaction Predictor (`pages/prediction.py`)
- **Dual-Model Inference Engine:**
  - **RandomForestClassifier (11 features):** Forecasts customer satisfaction tier (Low, Moderate, High) with confidence % and class probability distribution.
  - **RandomForestRegressor (11 features):** Predicts precise continuous **Satisfaction Score (1.0–5.0)**.
- **11-Feature Input Form:** Demographics, Transaction Economics, and Recency/Loyalty metrics.
- **Actionable Business Recommendation:** Dynamically generated customer intervention strategy.
- **Feature Importance Ranking:** Visualized relative predictor weights.

### 4. 🔵 Customer Segmentation (`pages/segmentation.py`)
- **KMeans (k=4)** clustering with `StandardScaler` normalisation across 6 behavioral dimensions (`Avg_Spend`, `Items_Purchased`, `Days_Since_Last_Purchase`, `Membership_Enc`, `Income_Enc`, `Age`).
- **4 Behavioral Personas:**

| Persona | Key Characteristics | Strategy |
|---|---|---|
| 🏆 **High-Value Champions** | Highest spend ($297+), highest basket (6.8 items), top satisfaction (3.83/5), 80%+ members | VIP previews, luxury bundles, concierge service |
| 🌟 **Engaged Members** | 100% membership, recent buyers (~20 days), steady satisfaction (3.28/5) | Tiered point multipliers, anniversary rewards |
| 🎯 **Prospect Shoppers** | 0% membership, recent buyers (~20 days), moderate spend ($178) | Sign-up incentives, welcome vouchers |
| ⚠️ **At-Risk Inactive** | Long dormancy (58+ days), lowest satisfaction (2.76/5) | Automated win-back campaigns, urgent retention discounts |

- Multi-dimensional visualisations: Spend vs Recency, Basket Size vs Spend, Category preference, and Income mix.
- **Segment Insights:** 4 responsive KPI cards displaying dynamically computed metrics for:
  1. Largest Customer Segment (`Engaged Members` — 3,097 customers, 31.0%)
  2. Highest Avg-Spending Segment (`High-Value Champions` — $297.21 avg spend)
  3. Most Inactive Segment (`At-Risk Inactive` — 58.5 days avg inactivity)
  4. Prospect Shoppers — Conversion Opportunity (`Prospect Shoppers` — 3,061 customers, 30.6%)

### 5. 💡 Strategic Business Insights (`pages/insights.py`)
- Executive intelligence summary:
  1. **Top Category & Spend Drivers:** Revenue leaders and cross-sell opportunities
  2. **Membership Program ROI:** Spend and satisfaction uplift from enrolled members
  3. **Discount Engagement:** Basket-building effectiveness of promotional pricing
  4. **Recency & Churn Risk:** Identifying critical dormancy thresholds before Day 30
  5. **Payment Method Dynamics:** Adoption and correlation with satisfaction
- **Actionable Strategic Recommendations Matrix:** Prioritized intervention cards with impact tags.

---

## 🛠️ Technology Stack

| Layer | Technologies & Capabilities |
|---|---|
| **Frontend** | Streamlit 1.58, Plotly Interactive Charts, Custom Dark Theme CSS |
| **Data Processing & Analytics** | Pandas 2.2, NumPy 2.1, Missing Value Validation & Categorical Encodings |
| **Machine Learning** | scikit-learn 1.5 (`RandomForestClassifier`, `RandomForestRegressor`, `KMeans`, `StandardScaler`) |
| **Deployment & Model Management** | Streamlit Community Cloud, Joblib 1.5 Persistence, Real-time Dashboard Inference |

---

## ⚙️ Running Locally

```bash
# 1. Clone repository
git clone https://github.com/Anshu926/Customer-Behavior-Analytics-Platform.git
cd Customer-Behavior-Analytics-Platform

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch application
streamlit run app.py
```

Access in your browser at **http://localhost:8501**.
