"""
src/train_model.py
End-to-end model training, cross-validation, hyperparameter tuning, evaluation, SHAP explainer setup, and persistence.
Uses Algorithmic Loss Weighting (scale_pos_weight & class_weight='balanced') to handle class imbalance.
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier, StackingClassifier
from xgboost import XGBClassifier
import shap

from src.preprocessing import load_raw_data, get_feature_columns, engineer_features, get_preprocessor
from src.evaluation import compute_metrics, get_confusion_matrix_data


def train_models_pipeline(data_path: str = "data/customer_churn.csv", verbose: bool = True):
    """
    Train and evaluate multiple models using Algorithmic Loss Weighting, perform hyperparameter tuning,
    setup SHAP explainer, and save trained artifacts to models/ directory.
    """
    if verbose:
        print("=== Step 1: Loading & Preprocessing Dataset ===")
    
    df_clean, X_raw, y = load_raw_data(data_path)
    X_eng = engineer_features(X_raw)
    
    # Stratified Train/Test Split (80/20)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_eng, y, test_size=0.2, random_state=42, stratify=y
    )
    
    cat_cols, num_cols = get_feature_columns(X_train_raw)
    
    # Fit preprocessor ONLY on training set to prevent data leakage
    preprocessor = get_preprocessor(cat_cols, num_cols)
    X_train_trans = preprocessor.fit_transform(X_train_raw)
    X_test_trans = preprocessor.transform(X_test_raw)
    
    feature_names = preprocessor.get_feature_names_out().tolist()
    
    # Algorithmic Loss Weighting ratio for XGBoost
    neg_count, pos_count = np.bincount(y_train)
    scale_pos = neg_count / max(pos_count, 1)
    
    if verbose:
        print(f"Training shape: {X_train_trans.shape}, Test shape: {X_test_trans.shape}")
        print(f"Class imbalance ratio (scale_pos_weight): {scale_pos:.2f} (Retained={neg_count}, Churned={pos_count})")
        print("\n=== Step 2: Defining ML Models with Algorithmic Loss Weighting ===")
        
    models = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5, min_samples_split=10, class_weight="balanced", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=4, class_weight="balanced", random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.7,
            scale_pos_weight=scale_pos, eval_metric="logloss", random_state=42
        ),
        "Soft Voting Ensemble": VotingClassifier(
            estimators=[
                ("lr", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)),
                ("rf", RandomForestClassifier(n_estimators=200, max_depth=8, class_weight="balanced", random_state=42)),
                ("xgb", XGBClassifier(n_estimators=200, max_depth=3, scale_pos_weight=scale_pos, eval_metric="logloss", random_state=42))
            ],
            voting="soft"
        ),
        "Stacking Ensemble (Max Accuracy)": StackingClassifier(
            estimators=[
                ("xgb", XGBClassifier(n_estimators=250, max_depth=3, learning_rate=0.03, scale_pos_weight=scale_pos, eval_metric="logloss", random_state=42)),
                ("rf", RandomForestClassifier(n_estimators=200, max_depth=8, class_weight="balanced", random_state=42)),
                ("et", ExtraTreesClassifier(n_estimators=200, max_depth=8, class_weight="balanced", random_state=42))
            ],
            final_estimator=LogisticRegression(C=0.1, class_weight="balanced", random_state=42),
            cv=5
        )
    }
    
    # 5-Fold Stratified Cross-Validation on training data
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results_summary = {}
    
    for name, model in models.items():
        if verbose:
            print(f"Running 5-fold CV for {name}...")
        scores = cross_validate(
            model, X_train_trans, y_train, cv=cv, scoring=["roc_auc", "f1"], n_jobs=-1
        )
        cv_results_summary[name] = {
            "ROC-AUC Mean": float(scores["test_roc_auc"].mean()),
            "ROC-AUC Std": float(scores["test_roc_auc"].std()),
            "F1 Mean": float(scores["test_f1"].mean()),
            "F1 Std": float(scores["test_f1"].std()),
        }
        
    if verbose:
        print("\n=== Step 3: Hyperparameter Tuning for XGBoost ===")
        
    xgb_param_grid = {
        "n_estimators": [100, 150, 200],
        "max_depth": [3, 4, 6],
        "learning_rate": [0.03, 0.05, 0.1],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0]
    }
    
    xgb_search = RandomizedSearchCV(
        estimator=XGBClassifier(scale_pos_weight=scale_pos, eval_metric="logloss", random_state=42),
        param_distributions=xgb_param_grid,
        n_iter=8,
        scoring="roc_auc",
        cv=cv,
        random_state=42,
        n_jobs=-1
    )
    xgb_search.fit(X_train_trans, y_train)
    best_xgb_model = xgb_search.best_estimator_
    models["XGBoost (Tuned)"] = best_xgb_model
    
    if verbose:
        print(f"Best XGBoost Params: {xgb_search.best_params_}")
        print("\n=== Step 4: Model Evaluation on Test Set ===")
        
    evaluation_records = []
    trained_fitted_models = {}
    
    for name, model in models.items():
        model.fit(X_train_trans, y_train)
        trained_fitted_models[name] = model
        
        y_pred = model.predict(X_test_trans)
        y_prob = model.predict_proba(X_test_trans)[:, 1] if hasattr(model, "predict_proba") else None
        
        metrics = compute_metrics(y_test, y_pred, y_prob)
        metrics["Model"] = name
        evaluation_records.append(metrics)
        
    metrics_df = pd.DataFrame(evaluation_records)
    cols_order = ["Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    metrics_df = metrics_df[cols_order].sort_values(by=["ROC-AUC", "F1-Score"], ascending=False).reset_index(drop=True)
    
    best_model_name = metrics_df.iloc[0]["Model"]
    best_model = trained_fitted_models[best_model_name]
    
    if verbose:
        print("\nModel Evaluation Comparison Table:")
        print(metrics_df.to_string(index=False))
        print(f"\nTop Performing Model based on ROC-AUC & F1: {best_model_name}")

    if verbose:
        print("\n=== Step 5: Setting Up SHAP Explainer ===")

    # Initialize SHAP TreeExplainer for tuned XGBoost model
    explainer = shap.TreeExplainer(best_xgb_model)
    test_sample = X_test_trans[:500]
    shap_values_test = explainer(test_sample)
    
    if verbose:
        print("=== Step 6: Saving Models & Metadata Artifacts ===")

    os.makedirs("models", exist_ok=True)
    
    # Save main churn pipeline & model bundle
    model_bundle = {
        "preprocessor": preprocessor,
        "models": trained_fitted_models,
        "best_model_name": best_model_name,
        "best_model": best_model,
        "xgboost_model": best_xgb_model,
        "feature_names": feature_names,
        "cat_cols": cat_cols,
        "num_cols": num_cols
    }
    joblib.dump(model_bundle, "models/churn_model.pkl")
    
    # Save detailed evaluation metadata & SHAP background
    metadata_bundle = {
        "metrics_df": metrics_df,
        "cv_results": cv_results_summary,
        "X_train_raw": X_train_raw,
        "X_test_raw": X_test_raw,
        "X_train_trans": X_train_trans,
        "X_test_trans": X_test_trans,
        "y_train": y_train,
        "y_test": y_test,
        "shap_explainer": explainer,
        "shap_values_test": shap_values_test,
        "test_sample": test_sample,
        "best_params_xgb": xgb_search.best_params_
    }
    joblib.dump(metadata_bundle, "models/model_metadata.pkl")

    if verbose:
        print("Model training pipeline completed successfully! Saved to models/ churn_model.pkl & model_metadata.pkl")

    return model_bundle, metadata_bundle


if __name__ == "__main__":
    train_models_pipeline()
