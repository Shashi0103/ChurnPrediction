"""
src/predict.py
Inference engine: loading models, single customer prediction, risk categorization, SHAP explanation generation, and batch predictions.
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import numpy as np
import pandas as pd
import shap

from src.preprocessing import engineer_features, load_raw_data


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def load_model_artifacts(model_path=None, metadata_path=None):
    """
    Load pre-trained model bundle and evaluation metadata bundle from disk.
    """
    if model_path is None:
        model_path = os.path.join(BASE_DIR, "models", "churn_model.pkl")
    if metadata_path is None:
        metadata_path = os.path.join(BASE_DIR, "models", "model_metadata.pkl")

    if not os.path.exists(model_path) or not os.path.exists(metadata_path):
        return None, None
    
    model_bundle = joblib.load(model_path)
    metadata_bundle = joblib.load(metadata_path)
    return model_bundle, metadata_bundle


def get_risk_category(prob: float) -> tuple[str, str]:
    """
    Map churn probability to risk category and display color badge.
    Thresholds:
      Low Risk: prob < 0.30
      Medium Risk: 0.30 <= prob <= 0.60
      High Risk: prob > 0.60
    """
    if prob < 0.30:
        return "LOW RISK", "#10b981"  # Emerald green
    elif prob <= 0.60:
        return "MEDIUM RISK", "#f59e0b"  # Amber orange
    else:
        return "HIGH RISK", "#ef4444"  # Red


def predict_single_customer(customer_dict: dict, model_bundle: dict, threshold: float = 0.50):
    """
    Predict churn probability, binary prediction, and risk level for a single customer.
    """
    df_raw = pd.DataFrame([customer_dict])
    df_eng = engineer_features(df_raw)
    
    preprocessor = model_bundle["preprocessor"]
    model = model_bundle["best_model"]
    
    X_trans = preprocessor.transform(df_eng)
    prob = float(model.predict_proba(X_trans)[0, 1])
    prediction = int(prob >= threshold)
    risk_label, risk_color = get_risk_category(prob)
    
    return {
        "churn_probability": prob,
        "prediction": prediction,
        "prediction_label": "Churn (Yes)" if prediction == 1 else "Retained (No)",
        "risk_label": risk_label,
        "risk_color": risk_color,
        "threshold": threshold,
        "X_trans": X_trans,
        "df_eng": df_eng
    }


def explain_single_customer(customer_dict: dict, model_bundle: dict, metadata_bundle: dict, top_n: int = 5, threshold: float = 0.50):
    """
    Compute SHAP values for an individual customer and extract top factors
    increasing and decreasing churn risk dynamically.
    """
    pred_res = predict_single_customer(customer_dict, model_bundle, threshold=threshold)
    X_trans = pred_res["X_trans"]
    
    explainer = metadata_bundle["shap_explainer"]
    feature_names = model_bundle["feature_names"]
    
    # Compute SHAP explanation for the single instance
    shap_exp_raw = explainer(X_trans)[0]
    
    # Clean feature names for readability
    clean_names = [f.replace("cat__", "").replace("num__", "") for f in feature_names]
    
    # Construct explanation object with clean feature names
    shap_exp = shap.Explanation(
        values=shap_exp_raw.values,
        base_values=shap_exp_raw.base_values,
        data=X_trans[0],
        feature_names=clean_names
    )
    
    # Extract feature impacts
    shap_values_arr = shap_exp.values
    df_factors = pd.DataFrame({
        "feature": clean_names,
        "shap_value": shap_values_arr,
        "abs_shap": np.abs(shap_values_arr),
        "value": X_trans[0]
    })
    
    risk_increasing = (
        df_factors[df_factors["shap_value"] > 0]
        .sort_values(by="shap_value", ascending=False)
        .head(top_n)
    )
    risk_decreasing = (
        df_factors[df_factors["shap_value"] < 0]
        .sort_values(by="shap_value", ascending=True)
        .head(top_n)
    )
    
    return {
        "pred_res": pred_res,
        "shap_exp": shap_exp,
        "risk_increasing": risk_increasing,
        "risk_decreasing": risk_decreasing,
        "df_factors": df_factors
    }


def predict_batch(df_input: pd.DataFrame, model_bundle: dict, threshold: float = 0.50):
    """
    Preprocess and generate predictions for a batch DataFrame.
    """
    df_clean = df_input.copy()
    
    # Handle TotalCharges if present
    if "TotalCharges" in df_clean.columns:
        df_clean["TotalCharges"] = pd.to_numeric(
            df_clean["TotalCharges"].replace(" ", np.nan), errors="coerce"
        )
        if "MonthlyCharges" in df_clean.columns and "tenure" in df_clean.columns:
            df_clean["TotalCharges"] = df_clean["TotalCharges"].fillna(
                df_clean["MonthlyCharges"] * df_clean["tenure"]
            )
            
    if "SeniorCitizen" in df_clean.columns and df_clean["SeniorCitizen"].dtype in [int, np.int64]:
        df_clean["SeniorCitizen"] = df_clean["SeniorCitizen"].map({1: "Yes", 0: "No"}).fillna("No")

    customer_ids = df_clean["customerID"].tolist() if "customerID" in df_clean.columns else [f"CUST_{i+1001}" for i in range(len(df_clean))]
    
    # Drop IDs/target if present in features
    drop_cols = [c for c in ["customerID", "Churn", "Churn_Num"] if c in df_clean.columns]
    X_raw = df_clean.drop(columns=drop_cols)
    X_eng = engineer_features(X_raw)
    
    preprocessor = model_bundle["preprocessor"]
    model = model_bundle["best_model"]
    
    X_trans = preprocessor.transform(X_eng)
    probs = model.predict_proba(X_trans)[:, 1]
    predictions = (probs >= threshold).astype(int)
    
    results = []
    for cid, prob, pred in zip(customer_ids, probs, predictions):
        risk_label, _ = get_risk_category(prob)
        results.append({
            "CustomerID": cid,
            "Churn Probability": round(prob, 4),
            "Prediction": "Yes" if pred == 1 else "No",
            "Risk Category": risk_label
        })
        
    return pd.DataFrame(results), X_trans
