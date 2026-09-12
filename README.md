# Customer Behavior Analytics Platform

An enterprise-grade Customer Intelligence and Behavior Analytics Platform built with **Streamlit**, **scikit-learn**, **Pandas**, and **Plotly**.

---

## 📊 Dataset Reference & Attribution

This project utilizes the **E-Commerce Customer Behavior Dataset** from Kaggle:

- **Dataset Name:** [E-Commerce Customer Behavior Dataset](https://www.kaggle.com/datasets/uom190346a/e-commerce-customer-behavior-dataset)
- **Source:** Kaggle
- **Author/Creator:** [@uom190346a](https://www.kaggle.com/uom190346a)
- **Files in Project:**
  - customer_data.csv: Raw dataset containing categorical and continuous customer attributes.
  - clean_customer_data.csv: Preprocessed and numerical-encoded dataset used for model training and clustering.

### Dataset Schema & Features:
| Feature | Description | Values / Types |
| :--- | :--- | :--- |
| Customer ID | Unique customer identifier | Integer (101 - 450) |
| Gender | Customer gender | Female, Male (0: Female, 1: Male) |
| Age | Customer age in years | Integer |
| City / City Tier | Location / Tier rank of customer city | 1, 2, 3 |
| Membership Type | Loyalty program tier | Bronze (1), Silver (2), Gold (3) |
| Total Spend | Total monetary spend | Float ($) |
| Items Purchased | Total quantity of items bought | Integer |
| Average Rating | Average review rating provided | Float (1.0 - 5.0) |
| Discount Applied | Whether a discount was applied | Boolean (0: No, 1: Yes) |
| Days Since Last Purchase | Recency indicator (days) | Integer |
| Satisfaction Level | Target customer satisfaction category | Unsatisfied (0), Neutral (1), Satisfied (2) |

---

## 🚀 Key Modules & Capabilities

1. **Business Dashboard (pages/dashboard.py)**:
   - Executive KPIs: Total Customers, Average Spend, Total Revenue, Average Rating.
   - Satisfaction level breakdown, City-tier spend patterns, and discount engagement metrics.

2. **Satisfaction Predictor (pages/prediction.py)**:
   - Machine Learning model powered by RandomForestClassifier.
   - Real-time customer satisfaction inference with confidence score and personalized business actions.

3. **Customer Segmentation (pages/segmentation.py)**:
   - Unsupervised clustering via KMeans + StandardScaler.
   - Identifies customer personas: High-Spender VIPs, Bargain Hunters, Regulars, and Churn-Risk customers.

4. **Strategic Insights (pages/insights.py)**:
   - Automated business recommendations, feature importance rankings, and churn prevention strategies.

---

## 🛠️ How to Run the Application

1. **Install Dependencies:**
```bash
pip install -r requirements.txt
```

2. **Start the Streamlit Application:**
```bash
streamlit run app.py
```

3. Open your browser at **http://localhost:8501**.
