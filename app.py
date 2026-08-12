"""
app.py
Streamlit Web Application: Customer Churn Prediction Platform (Minimalist 4-Page Architecture)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Setup page configuration
st.set_page_config(
    page_title="Customer Churn Prediction Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    /* Main Background & Card Styling */
    .stApp {
        background-color: #0f172a;
        color: #f8fafc;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #6366f1;
        margin-bottom: 4px;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .status-badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1rem;
        color: #ffffff;
        text-align: center;
    }
    
    .factor-item-increase {
        background-color: rgba(239, 68, 68, 0.1);
        border-left: 4px solid #ef4444;
        padding: 10px 14px;
        border-radius: 4px;
        margin-bottom: 8px;
        color: #fca5a5;
    }
    
    .factor-item-decrease {
        background-color: rgba(16, 185, 129, 0.1);
        border-left: 4px solid #10b981;
        padding: 10px 14px;
        border-radius: 4px;
        margin-bottom: 8px;
        color: #6ee7b7;
    }
    
    .sidebar-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #6366f1;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

# Import internal modules
from src.preprocessing import load_raw_data, engineer_features, get_feature_columns
from src.train_model import train_models_pipeline
from src.predict import load_model_artifacts, predict_single_customer, explain_single_customer, predict_batch
from src.evaluation import (
    plot_confusion_matrix_fig, plot_roc_curves_fig, plot_feature_importance_fig,
    plot_shap_summary_fig, plot_shap_bar_fig, plot_individual_shap_waterfall_fig
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Caching dataset loading
@st.cache_data
def get_cached_data():
    data_path = os.path.join(BASE_DIR, "data", "customer_churn.csv")
    return load_raw_data(data_path)

# Caching models loading
@st.cache_resource
def get_cached_models():
    model_bundle, metadata_bundle = load_model_artifacts()
    if model_bundle is None:
        model_bundle, metadata_bundle = train_models_pipeline(verbose=False)
    return model_bundle, metadata_bundle

# Sidebar Navigation (Minimalist 4-Page Layout)
logo_path = os.path.join(BASE_DIR, "assets", "logo.png")
st.sidebar.image(logo_path if os.path.exists(logo_path) else "📊", use_container_width=True)
st.sidebar.markdown("<div class='sidebar-header'>Navigation</div>", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Select Module",
    options=[
        "📊 Dashboard & Data Insights",
        "📈 Model Performance & SHAP",
        "👤 Predict Customer Churn",
        "📂 Batch CSV Prediction"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Engine:** Tuned XGBoost + SHAP")
st.sidebar.markdown("**Status:** 🟢 Models Active")

with st.sidebar.expander("ℹ️ About System Architecture"):
    st.markdown("""
    * **Preprocessing**: Standardized Scaling & One-Hot Encoding fitted strictly on 80% training split.
    * **Validation**: 5-Fold Stratified Cross Validation.
    * **Tuning**: Automated `RandomizedSearchCV`.
    * **XAI**: SHAP `TreeExplainer` providing additive feature attributions.
    """)

# Load cached data and models
df_clean, X_raw, y_target = get_cached_data()
model_bundle, metadata_bundle = get_cached_models()

if df_clean is None or y_target is None:
    st.error("Failed to initialize dataset. Please check data/customer_churn.csv.")
    st.stop()


# ==========================================
# 📊 PAGE 1: DASHBOARD & DATA INSIGHTS
# ==========================================
if page == "📊 Dashboard & Data Insights":
    st.title("📊 Customer Churn Dashboard & Data Insights")
    st.markdown("Executive summary of customer demographics, churn indicators, and key feature distributions.")
    
    total_cust = len(df_clean)
    churn_cust = int(((y_target == 1) | (y_target == "Yes")).sum())
    churn_rate = (churn_cust / total_cust) * 100 if total_cust > 0 else 0.0
    avg_monthly = df_clean["MonthlyCharges"].mean()
    avg_tenure = df_clean["tenure"].mean()
    
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    kpi1.metric("Total Customers", f"{total_cust:,}")
    kpi2.metric("Churned Customers", f"{churn_cust:,}")
    kpi3.metric("Churn Rate", f"{churn_rate:.2f}%")
    kpi4.metric("Avg Monthly Charge", f"${avg_monthly:.2f}")
    kpi5.metric("Avg Tenure", f"{avg_tenure:.1f} mos")
    
    st.markdown("---")
    
    # Dataset Exploratory Visualizations
    st.subheader("Data Exploratory Analysis & Feature Visualizations")
    v_tabs = st.tabs([
        "Churn Distribution", "Contract & Payment", "Tenure vs Churn", 
        "Monthly Charges", "Correlation Matrix"
    ])
    
    with v_tabs[0]:
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            df_clean["Churn"].value_counts().plot.pie(autopct="%1.1f%%", colors=["#10b981", "#ef4444"], explode=[0, 0.08], ax=ax)
            ax.set_ylabel("")
            ax.set_title("Overall Churn Proportion", fontweight="bold")
            st.pyplot(fig)
        with c2:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            sns.countplot(data=df_clean, x="gender", hue="Churn", palette=["#10b981", "#ef4444"], ax=ax)
            ax.set_title("Churn Distribution by Gender", fontweight="bold")
            ax.grid(True, linestyle="--", alpha=0.3, axis="y")
            st.pyplot(fig)

    with v_tabs[1]:
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            sns.countplot(data=df_clean, x="Contract", hue="Churn", palette=["#10b981", "#ef4444"], ax=ax)
            ax.set_title("Churn by Contract Type", fontweight="bold")
            ax.grid(True, linestyle="--", alpha=0.3, axis="y")
            st.pyplot(fig)
        with c2:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            sns.countplot(data=df_clean, y="PaymentMethod", hue="Churn", palette=["#10b981", "#ef4444"], ax=ax)
            ax.set_title("Churn by Payment Method", fontweight="bold")
            ax.grid(True, linestyle="--", alpha=0.3, axis="x")
            st.pyplot(fig)
            
    with v_tabs[2]:
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            sns.kdeplot(data=df_clean, x="tenure", hue="Churn", common_norm=False, palette=["#10b981", "#ef4444"], fill=True, alpha=0.4, ax=ax)
            ax.set_title("Tenure Density Distribution", fontweight="bold")
            ax.grid(True, linestyle="--", alpha=0.3)
            st.pyplot(fig)
        with c2:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            sns.boxplot(data=df_clean, x="Churn", y="tenure", hue="Churn", palette=["#10b981", "#ef4444"], legend=False, ax=ax)
            ax.set_title("Tenure Boxplot by Churn Status", fontweight="bold")
            ax.grid(True, linestyle="--", alpha=0.3, axis="y")
            st.pyplot(fig)

    with v_tabs[3]:
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            sns.histplot(data=df_clean, x="MonthlyCharges", kde=True, color="#6366f1", ax=ax)
            ax.set_title("Monthly Charges Distribution", fontweight="bold")
            ax.grid(True, linestyle="--", alpha=0.3)
            st.pyplot(fig)
        with c2:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            sns.boxplot(data=df_clean, x="Churn", y="MonthlyCharges", hue="Churn", palette=["#10b981", "#ef4444"], legend=False, ax=ax)
            ax.set_title("Monthly Charges by Churn Status", fontweight="bold")
            ax.grid(True, linestyle="--", alpha=0.3, axis="y")
            st.pyplot(fig)

    with v_tabs[4]:
        st.markdown("#### Numerical Features Correlation Matrix")
        num_df = pd.DataFrame()
        num_df["tenure"] = pd.to_numeric(df_clean["tenure"], errors="coerce")
        num_df["MonthlyCharges"] = pd.to_numeric(df_clean["MonthlyCharges"], errors="coerce")
        num_df["TotalCharges"] = pd.to_numeric(df_clean["TotalCharges"], errors="coerce")
        num_df["Churn_Num"] = (df_clean["Churn"] == "Yes").astype(int) if "Churn" in df_clean.columns else pd.to_numeric(y_target, errors="coerce")
        
        corr_df = num_df.corr()
        fig, ax = plt.subplots(figsize=(6, 3.5))
        sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, ax=ax)
        ax.set_title("Numerical Features Correlation", fontweight="bold")
        st.pyplot(fig)


# ==========================================
# 📈 PAGE 2: MODEL PERFORMANCE & SHAP
# ==========================================
elif page == "📈 Model Performance & SHAP":
    st.title("📈 Model Performance & SHAP Explainability")
    st.markdown("Detailed model benchmark comparisons, cross-validation metrics, and global SHAP feature importance.")
    
    st.subheader("Model Evaluation Comparison Table (Test Set)")
    metrics_df = metadata_bundle["metrics_df"]
    st.dataframe(
        metrics_df.style.highlight_max(axis=0, subset=["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"], color="#1e3a8a"),
        use_container_width=True
    )
    
    st.info(f"🏆 **Top Selected Model**: `{model_bundle['best_model_name']}` selected based on high ROC-AUC and Recall trade-off.")
    
    st.markdown("---")
    
    m_tabs = st.tabs(["ROC Curves", "Confusion Matrix", "5-Fold CV & Tuning", "Global SHAP XAI"])
    
    with m_tabs[0]:
        fig_roc = plot_roc_curves_fig(model_bundle["models"], metadata_bundle["X_test_trans"], metadata_bundle["y_test"])
        st.pyplot(fig_roc)
        
    with m_tabs[1]:
        selected_model_name = st.selectbox("Select Model for Confusion Matrix:", options=list(model_bundle["models"].keys()), index=0)
        sel_model = model_bundle["models"][selected_model_name]
        y_pred = sel_model.predict(metadata_bundle["X_test_trans"])
        fig_cm = plot_confusion_matrix_fig(metadata_bundle["y_test"].values, y_pred, title=f"Confusion Matrix: {selected_model_name}")
        st.pyplot(fig_cm)
        
    with m_tabs[2]:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 5-Fold Stratified Cross-Validation Summary")
            cv_df = pd.DataFrame(metadata_bundle["cv_results"]).T
            st.dataframe(cv_df.style.format("{:.4f}"), use_container_width=True)
        with c2:
            st.markdown("#### Best Hyperparameters (XGBoost)")
            st.json(metadata_bundle["best_params_xgb"])
            if st.button("🔄 Retrain All Models Live"):
                with st.spinner("Retraining pipeline models..."):
                    train_models_pipeline(verbose=False)
                    st.cache_resource.clear()
                    st.cache_data.clear()
                    st.success("Models retrained successfully!")

    with m_tabs[3]:
        st.markdown("#### Global SHAP Feature Importance")
        shap_val_test = metadata_bundle["shap_values_test"]
        X_test_sample = metadata_bundle["test_sample"]
        feat_names = model_bundle["feature_names"]
        
        c1, c2 = st.columns(2)
        with c1:
            fig_shap_bar = plot_shap_bar_fig(shap_val_test, feat_names, max_display=12)
            st.pyplot(fig_shap_bar)
        with c2:
            fig_shap_sum = plot_shap_summary_fig(shap_val_test, X_test_sample, feat_names, max_display=12)
            st.pyplot(fig_shap_sum)


# ==========================================
# 👤 PAGE 3: PREDICT CUSTOMER CHURN
# ==========================================
elif page == "👤 Predict Customer Churn":
    st.title("👤 Predict Churn for Individual Customer")
    st.markdown("Enter customer information to calculate real-time churn probability, risk level, and SHAP factor breakdown.")
    
    with st.form("single_predict_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### Demographics")
            gender = st.selectbox("Gender", ["Female", "Male"])
            SeniorCitizen = st.selectbox("Senior Citizen", ["No", "Yes"])
            Partner = st.selectbox("Partner", ["No", "Yes"])
            Dependents = st.selectbox("Dependents", ["No", "Yes"])
            tenure = st.slider("Tenure (Months)", min_value=0, max_value=72, value=2)
            
        with col2:
            st.markdown("#### Services")
            PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
            MultipleLines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
            InternetService = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
            OnlineSecurity = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            OnlineBackup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
            DeviceProtection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
            TechSupport = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
            StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
            StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

        with col3:
            st.markdown("#### Account & Billing")
            Contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
            PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
            PaymentMethod = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
            ])
            MonthlyCharges = st.number_input("Monthly Charges ($)", min_value=18.0, max_value=150.0, value=95.5)
            TotalCharges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=191.0)
            threshold = st.slider("🎯 Decision Cutoff Threshold", min_value=0.10, max_value=0.90, value=0.50, step=0.05, help="Lower cutoff (<0.50) catches more churners (higher Recall); Higher cutoff (>0.50) reduces false alarms (higher Precision).")

        submit_btn = st.form_submit_button("🔮 Predict Customer Churn", use_container_width=True)
        
    if submit_btn:
        cust_dict = {
            "gender": gender,
            "SeniorCitizen": SeniorCitizen,
            "Partner": Partner,
            "Dependents": Dependents,
            "tenure": tenure,
            "PhoneService": PhoneService,
            "MultipleLines": MultipleLines,
            "InternetService": InternetService,
            "OnlineSecurity": OnlineSecurity,
            "OnlineBackup": OnlineBackup,
            "DeviceProtection": DeviceProtection,
            "TechSupport": TechSupport,
            "StreamingTV": StreamingTV,
            "StreamingMovies": StreamingMovies,
            "Contract": Contract,
            "PaperlessBilling": PaperlessBilling,
            "PaymentMethod": PaymentMethod,
            "MonthlyCharges": MonthlyCharges,
            "TotalCharges": TotalCharges
        }
        
        explanation = explain_single_customer(cust_dict, model_bundle, metadata_bundle, threshold=threshold)
        pred_res = explanation["pred_res"]
        
        st.markdown("---")
        st.subheader("Prediction Results")
        
        res_col1, res_col2, res_col3 = st.columns(3)
        res_col1.metric("Churn Probability", f"{pred_res['churn_probability']*100:.1f}%")
        res_col2.metric("Prediction Class", pred_res['prediction_label'])
        
        with res_col3:
            st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
            st.markdown(
                f"<div class='status-badge' style='background-color: {pred_res['risk_color']};'>{pred_res['risk_label']}</div>",
                unsafe_allow_html=True
            )

        st.markdown("---")
        st.subheader("Factors Contributing to Prediction (SHAP Explainability)")
        
        exp_col1, exp_col2 = st.columns(2)
        
        with exp_col1:
            st.markdown("#### ⬆️ Factors Increasing Churn Risk")
            for idx, row in explanation["risk_increasing"].iterrows():
                feat_clean = row['feature'].replace('cat__', '').replace('num__', '')
                st.markdown(
                    f"<div class='factor-item-increase'><b>{feat_clean}</b>: +{row['shap_value']:.3f} SHAP impact</div>",
                    unsafe_allow_html=True
                )
                
        with exp_col2:
            st.markdown("#### ⬇️ Factors Decreasing Churn Risk")
            for idx, row in explanation["risk_decreasing"].iterrows():
                feat_clean = row['feature'].replace('cat__', '').replace('num__', '')
                st.markdown(
                    f"<div class='factor-item-decrease'><b>{feat_clean}</b>: {row['shap_value']:.3f} SHAP impact</div>",
                    unsafe_allow_html=True
                )
                
        st.markdown("#### Individual Customer SHAP Waterfall Plot")
        fig_waterfall = plot_individual_shap_waterfall_fig(explanation["shap_exp"])
        st.pyplot(fig_waterfall)


# ==========================================
# 📂 PAGE 4: BATCH CSV PREDICTION
# ==========================================
elif page == "📂 Batch CSV Prediction":
    st.title("📂 Batch Customer Churn Prediction")
    st.markdown("Upload a CSV file of customers to generate bulk predictions, risk tiers, and downloadable reports.")
    
    uploaded_file = st.file_uploader("Upload Customer CSV File", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            st.write(f"Uploaded Dataset Preview ({len(df_upload)} rows):")
            st.dataframe(df_upload.head(5), use_container_width=True)
            
            if st.button("🚀 Run Batch Churn Predictions"):
                with st.spinner("Processing customer records through preprocessing pipeline..."):
                    results_df, X_batch_trans = predict_batch(df_upload, model_bundle)
                    st.success(f"Batch prediction completed for {len(results_df)} customers!")
                    
                    st.dataframe(results_df, use_container_width=True)
                    
                    csv_data = results_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Batch Predictions CSV Report",
                        data=csv_data,
                        file_name="churn_predictions_report.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
        except Exception as e:
            st.error(f"Error processing uploaded CSV file: {e}")
    else:
        st.info("No CSV uploaded yet. Download a sample batch template below to test predictions.")
        sample_batch = df_clean.drop(columns=["Churn", "Churn_Num"]).head(10)
        sample_csv = sample_batch.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📄 Download Sample Batch CSV Template",
            data=sample_csv,
            file_name="sample_customer_batch.csv",
            mime="text/csv"
        )
