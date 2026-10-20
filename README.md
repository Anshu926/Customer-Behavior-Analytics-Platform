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
├── app.py                     # Landing page / home & dataset overview
├── utils.py                   # Data loaders, ML pipeline, design system & styling
├── requirements.txt           # Pinned dependencies
│
├── pages/
│   ├── dashboard.py           # Business dashboard, KPIs & behavioral charts
│   ├── prediction.py          # Satisfaction predictor (RandomForest - 11 features)
│   ├── segmentation.py        # Customer segmentation (KMeans - 4 personas)
│   └── insights.py            # Automated diagnostic business insights
│
├── customer_data.csv          # Raw 10,000-row dataset (14 columns)
├── clean_customer_data.csv    # Preprocessed & encoded dataset (24 columns)
│
├── predictor.pkl              # Trained RandomForestClassifier model
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

## 🚀 Platform Modules

### 1. 🏠 Home Page (`app.py`)
- Hero banner with enterprise dark aesthetic
- Live KPI ribbon: **Total Customers, Average Spend, Total Revenue, Avg Satisfaction**
- Dataset snapshot: Membership enrollment, satisfaction tier breakdown, average items per transaction, recency interval
- Architecture and technology stack cards

### 2. 📊 Business Dashboard (`pages/dashboard.py`)
- **6 Executive KPI Cards:** Total Customers, Avg Spend, Total Revenue, Avg Satisfaction, Avg Basket Size, Membership Rate
- **Satisfaction Analysis:** Donut chart & continuous score distribution histogram
- **Category Dynamics:** Customer volume & average spend by category
- **Shopping Behavioral Dynamics:** Basket size vs spend, discount elasticity analysis, recency cohorts
- **Geographic & Economic Breakdown:** Regional performance and income level satisfaction
- **Demographics & Payment Methods:** Payment preferences & age/gender distributions
- **Correlation Engine:** Spend vs satisfaction scatter with trendline

### 3. 🎯 Satisfaction Predictor (`pages/prediction.py`)
- **RandomForestClassifier** trained on 10,000 records with 11 behavioral and demographic features
- **11-Feature Input Form:**
  - Demographics: Age, Gender, Income Level, Region
  - Transactions: Category, Avg Spend, Items Purchased, Payment Mode
  - Engagement: Membership Status, Discount Applied, Days Since Last Purchase
- **Real-time Inference Output:**
  - Predicted satisfaction class (Low / Moderate / High)
  - Confidence percentage
  - Probability distribution across all three classes
  - Actionable business recommendation tailored to the outcome
  - Feature importance ranking

### 4. 🔵 Customer Segmentation (`pages/segmentation.py`)
- **KMeans (k=4)** clustering with StandardScaler normalisation on monetary, recency, basket size, income, and loyalty features
- **4 Behavioral Personas:**

| Persona | Key Characteristics | Strategy |
|---|---|---|
| 🏆 **High-Value Champions** | Highest spend ($297+), highest basket (6.8 items), top satisfaction (3.83/5), 80%+ members | VIP previews, luxury bundles, concierge service |
| 🌟 **Engaged Members** | 100% membership, recent buyers (~20 days), steady satisfaction (3.28/5) | Tiered point multipliers, anniversary rewards |
| 🎯 **Prospect Shoppers** | 0% membership, recent buyers (~20 days), moderate spend ($178) | Sign-up incentives, welcome vouchers |
| ⚠️ **At-Risk Inactive** | Long dormancy (58+ days), lowest satisfaction (2.76/5) | Automated win-back campaigns, urgent retention discounts |

- Multi-dimensional visualisations (Spend vs Recency, Basket Size vs Spend, Category preference, Income mix)
- **Live Persona Classifier:** Instant classification and targeted marketing recommendation for new customer inputs.

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

| Layer | Tools |
|---|---|
| **Frontend** | Streamlit 1.53+, Plotly Interactive Charts, Custom Dark Enterprise CSS |
| **Backend & Pipeline** | Python 3.9+, Pandas 2.2, NumPy 2.1 |
| **Machine Learning** | scikit-learn 1.5 (RandomForestClassifier, KMeans, StandardScaler) |
| **Model Persistence** | Joblib 1.5 |
| **Deployment** | Streamlit Community Cloud |

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
