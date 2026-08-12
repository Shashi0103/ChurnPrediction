# Customer Churn Prediction Platform

An end-to-end, production-quality **Data Science, Machine Learning, and Explainable AI (XAI) application** built with Python, Scikit-Learn, XGBoost, SHAP, and Streamlit.

![Platform Banner](assets/logo.png)

---

## 📌 Project Overview

Customer churn occurs when customers discontinue their service relationship with a business. Retaining existing customers is significantly more cost-effective than acquiring new ones.

This project delivers a complete enterprise-level workflow that predicts customer churn risk using machine learning while providing **transparent, individual feature attributions via SHAP (Shapley Additive exPlanations)**.

### Key Objectives
* **Predictive Accuracy**: Train and compare baseline linear models, tree models, and tuned gradient boosting algorithms.
* **Model Explainability**: Deconstruct complex XGBoost predictions into intuitive, human-understandable factor contributions using SHAP.
* **Interactive Analytics**: Provide executive dashboards, exploratory data analysis, real-time single-customer risk scoring, and bulk CSV batch processing through a modern Streamlit web interface.

---

## ✨ Features

* **🏠 Home Landing Page**: Executive summary, project pillars, core architecture cards.
* **📊 Customer Risk Dashboard**: Real-time KPI metrics (Total Customers, Churn %, Avg Monthly Charges, Avg Tenure) and interactive distribution charts.
* **🔎 Exploratory Data Analysis (EDA)**: 10 structured visualizations covering demographic, service, and billing patterns alongside a numerical correlation matrix heatmap.
* **🤖 Automated Model Training**: 80/20 stratified split, 5-Fold Stratified Cross-Validation, and automated hyperparameter tuning using `RandomizedSearchCV`.
* **📈 Model Evaluation Suite**: Performance comparison matrix, interactive Confusion Matrices, ROC curve overlays, and metric trade-off analysis.
* **🧠 SHAP Explainability**: Global feature importance summary plots, mean absolute SHAP value bar charts, and feature contribution rules.
* **👤 Real-Time Predict Churn**: Single customer form with real-time churn probability, risk category badges (Low <30%, Medium 30-60%, High >60%), dynamic SHAP factor lists, and interactive waterfall charts.
* **📂 Batch Prediction**: CSV file processing, automated feature transformation, bulk churn risk scoring, and downloadable prediction reports.
* **ℹ️ Technical Documentation**: Architectural diagrams, technology stack, and methodological disclaimers.

---

## 🔄 Machine Learning Workflow

```text
Data Collection (IBM Telco Dataset)
      ↓
Data Cleaning & Null Handling
      ↓
Exploratory Data Analysis (EDA)
      ↓
Domain Feature Engineering
      ↓
Preprocessing Pipeline (OneHotEncoder + StandardScaler)
      ↓
Stratified Train/Test Split (80/20)
      ↓
Model Training (Logistic Regression, Decision Tree, Random Forest, XGBoost)
      ↓
5-Fold Stratified Cross-Validation
      ↓
Hyperparameter Tuning (RandomizedSearchCV)
      ↓
Model Evaluation & Selection (ROC-AUC & F1 Focus)
      ↓
XGBoost Final Selection
      ↓
SHAP TreeExplainer Initialization
      ↓
Individual & Batch Churn Risk Prediction + Waterfall XAI
```

---

## 🤖 Machine Learning Algorithms

1. **Logistic Regression**: Linear baseline classifier evaluated with `class_weight='balanced'`.
2. **Decision Tree**: Simple rule-based classifier regularized to prevent overfitting (`max_depth=5`, `min_samples_split=10`).
3. **Random Forest**: Ensemble of 100 decision trees utilizing random feature sub-sampling (`n_estimators=100`, `max_depth=8`).
4. **XGBoost (Tuned)**: Gradient-boosted decision trees optimized with `scale_pos_weight` and hyperparameter tuned via `RandomizedSearchCV` (`subsample=0.7`, `n_estimators=200`, `max_depth=3`, `learning_rate=0.03`).

---

## 🧠 Explainable AI (SHAP)

### What is SHAP?
SHAP (Shapley Additive exPlanations) is a game-theoretic approach to explain the output of any machine learning model. It connects optimal credit allocation with local explanations using Shapley values from cooperative game theory.

### Why SHAP is Used in This Project
Traditional feature importance methods (e.g., Gini importance in Decision Trees) show global importance but cannot explain **individual customer predictions**. SHAP provides additive feature attributions for every single customer.

### Interpretation Rules
* **Positive SHAP Value (`+`)**: Pushes the model prediction **towards customer churn** (Increases Risk).
* **Negative SHAP Value (`-`)**: Pushes the model prediction **away from customer churn** (Decreases Risk).
* **Magnitude**: Reflects the strength of feature impact on the final churn probability.

> ⚠️ **Important Note on Causality**: SHAP explains **model behavior and feature attributions**, not physical causation. For example, a high SHAP value for `Contract_Month-to-month` indicates that month-to-month billing strongly contributed to the model's high churn prediction for that customer, not that changing contract terms will automatically guarantee retention.

---

## 📊 Model Evaluation & Results

All models were evaluated on an independent 20% test dataset (1,409 customers) using 80% stratified training.

### Test Set Performance Comparison Table

| Model | Imbalance Handling | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **XGBoost (Tuned)** 🏆 | `scale_pos_weight = 2.77` | **0.7452** | **0.5129** | **0.7968** | **0.6241** | **0.8466** |
| **Logistic Regression** | `class_weight='balanced'` | 0.7445 | 0.5119 | 0.8048 | 0.6258 | 0.8450 |
| **Stacking Ensemble** 🚀 | `class_weight='balanced'` | 0.7566 | 0.5274 | 0.7968 | 0.6347 | 0.8432 |
| **Soft Voting Ensemble** | Soft Averaging | 0.7608 | 0.5352 | 0.7513 | 0.6251 | 0.8419 |
| **Random Forest** | `class_weight='balanced'` | 0.7630 | 0.5376 | 0.7647 | 0.6313 | 0.8410 |
| **Decision Tree** | `class_weight='balanced'` | 0.7417 | 0.5085 | 0.7995 | 0.6216 | 0.8297 |

* **Class Imbalance Strategy**: Uses pure **Algorithmic Loss Weighting** (`scale_pos_weight` in XGBoost and `class_weight='balanced'` in scikit-learn models), penalizing misclassifications on minority churn samples without creating noisy synthetic data.

---

## 📈 Key SHAP Global Findings

Across the IBM Telco dataset, the top 5 features driving customer churn risk are:

1. **`Contract_Month-to-month`**: Short-term contracts are the single strongest positive predictor of churn.
2. **`tenure`**: Newer customers (short tenure < 12 months) display significantly higher churn probabilities.
3. **`InternetService_Fiber optic`**: Customers with Fiber Optic service experience higher churn rates, likely driven by competitive pricing or service sensitivity.
4. **`OnlineSecurity_No`**: Absence of online security services correlates with increased churn risk.
5. **`MonthlyCharges`**: High recurring monthly fees push predictions toward higher churn risk.

---

## ⚙️ Installation & Usage Instructions

### Prerequisites
* Python 3.11+
* Git

### Step-by-Step Setup

```bash
# 1. Clone the repository
git clone <repository-url>
cd customer-churn-prediction

# 2. Create and activate a Python virtual environment
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate

# 3. Install required dependencies
pip install -r requirements.txt

# 4. Train models and generate serialized artifacts (Optional, automated on first run)
python src/train_model.py

# 5. Launch the Streamlit application
streamlit run app.py
```

The application will launch automatically in your browser at `http://localhost:8501`.

---

## 📁 Repository Structure

```text
customer-churn-prediction/
│
├── app.py                     # Main Streamlit Web Application
├── requirements.txt           # Python dependencies
├── README.md                  # Detailed documentation
│
├── data/
│   └── customer_churn.csv     # IBM Telco Customer Churn dataset
│
├── models/
│   ├── churn_model.pkl        # Serialized pipeline & trained model bundle
│   └── model_metadata.pkl     # Evaluation metrics & SHAP background data
│
├── src/
│   ├── preprocessing.py       # Data cleaning, feature engineering, and ColumnTransformer
│   ├── train_model.py         # Train/test split, 5-fold CV, tuning, & saving
│   ├── predict.py             # Single prediction, risk tiers, & batch inference
│   └── evaluation.py          # Metric calculations, confusion matrices, ROC, & SHAP plots
│
└── assets/
    └── logo.png               # Platform logo graphic
```

---

## 🚀 Future Roadmap & Enhancements

* **Probability Calibration**: Implement Platt Scaling or Isotonic Regression for calibrated probability outputs.
* **Customer Segmentation**: Integrate unsupervised K-Means clustering to group churn risks into actionable persona segments.
* **Cloud Deployment**: Containerize with Docker and deploy to Streamlit Community Cloud or AWS ECS.
* **Automated Data Drift Detection**: Monitor input feature drift using Evidently AI or Evidently Cloud.
