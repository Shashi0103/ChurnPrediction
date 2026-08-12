"""
src/preprocessing.py
Data loading, cleaning, feature engineering, and scikit-learn preprocessing pipeline.
"""

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_raw_data(data_path: str = "data/customer_churn.csv"):
    """
    Load customer churn CSV dataset, perform data cleaning and type conversion.
    """
    df = pd.read_csv(data_path)
    df_clean = df.copy()

    # Handle blank spaces in TotalCharges
    if "TotalCharges" in df_clean.columns:
        df_clean["TotalCharges"] = pd.to_numeric(
            df_clean["TotalCharges"].replace(" ", np.nan), errors="coerce"
        )
        df_clean["TotalCharges"] = df_clean["TotalCharges"].fillna(
            df_clean["MonthlyCharges"] * df_clean["tenure"]
        )

    if "SeniorCitizen" in df_clean.columns:
        df_clean["SeniorCitizen"] = df_clean["SeniorCitizen"].map({1: "Yes", 0: "No"}).fillna("No")

    if "Churn" in df_clean.columns:
        if df_clean["Churn"].dtype == object:
            df_clean["Churn_Num"] = df_clean["Churn"].map({"Yes": 1, "No": 0})
        else:
            df_clean["Churn_Num"] = df_clean["Churn"]
    else:
        df_clean["Churn_Num"] = 0

    drop_cols = [c for c in ["customerID", "Churn", "Churn_Num"] if c in df_clean.columns]
    X = df_clean.drop(columns=drop_cols)
    y = df_clean["Churn_Num"]

    return df_clean, X, y


def get_feature_columns(X: pd.DataFrame):
    """
    Categorize columns into numerical and categorical feature lists.
    """
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    return categorical_cols, numerical_cols


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer domain-specific features for maximum predictive accuracy.
    """
    df_feat = df.copy()
    
    # 1. Total active optional services count
    service_cols = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection", 
        "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    existing_services = [col for col in service_cols if col in df_feat.columns]
    if existing_services:
        df_feat["ServiceCount"] = (df_feat[existing_services] == "Yes").sum(axis=1)

    # 2. Tenure group categories
    if "tenure" in df_feat.columns:
        df_feat["TenureGroup"] = pd.cut(
            df_feat["tenure"], 
            bins=[-1, 6, 12, 24, 48, 60, 100], 
            labels=["0-6m", "6-12m", "12-24m", "24-48m", "48-60m", "60m+"]
        ).astype(str)

    # 3. Monthly charges ratio relative to total charges per tenure
    if "MonthlyCharges" in df_feat.columns and "TotalCharges" in df_feat.columns and "tenure" in df_feat.columns:
        expected_total = df_feat["MonthlyCharges"] * df_feat["tenure"]
        df_feat["ChargeDifference"] = df_feat["TotalCharges"] - expected_total
        df_feat["Avg_Monthly_Ratio"] = df_feat["TotalCharges"] / (df_feat["tenure"] * df_feat["MonthlyCharges"] + 1.0)

    # 4. Domain Interaction Indicators
    if "InternetService" in df_feat.columns:
        df_feat["Has_Fiber"] = (df_feat["InternetService"] == "Fiber optic").map({True: "Yes", False: "No"})

    if "Contract" in df_feat.columns:
        df_feat["Is_MonthToMonth"] = (df_feat["Contract"] == "Month-to-month").map({True: "Yes", False: "No"})

    if "PaymentMethod" in df_feat.columns:
        df_feat["Is_ElectronicCheck"] = (df_feat["PaymentMethod"] == "Electronic check").map({True: "Yes", False: "No"})

    if "OnlineSecurity" in df_feat.columns and "TechSupport" in df_feat.columns:
        df_feat["No_Security_Or_Support"] = (
            (df_feat["OnlineSecurity"] == "No") & (df_feat["TechSupport"] == "No")
        ).map({True: "Yes", False: "No"})

    if "SeniorCitizen" in df_feat.columns and "Partner" in df_feat.columns and "Dependents" in df_feat.columns:
        df_feat["Senior_Alone"] = (
            (df_feat["SeniorCitizen"] == "Yes") & (df_feat["Partner"] == "No") & (df_feat["Dependents"] == "No")
        ).map({True: "Yes", False: "No"})

    return df_feat


def get_preprocessor(categorical_cols: list, numerical_cols: list) -> ColumnTransformer:
    """
    Create a ColumnTransformer pipeline for categorical (OneHotEncoder) 
    and numerical (StandardScaler) features.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ],
        remainder="passthrough"
    )
    return preprocessor
