"""
src/evaluation.py
Model evaluation metrics, confusion matrix, ROC curve plotting, and SHAP visualization utilities.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
import shap


def compute_metrics(y_true, y_pred, y_prob):
    """
    Compute standard evaluation metrics: Accuracy, Precision, Recall, F1, and ROC-AUC.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    if y_prob is not None:
        auc = roc_auc_score(y_true, y_prob)
    else:
        auc = 0.5
        
    return {
        "Accuracy": float(acc),
        "Precision": float(prec),
        "Recall": float(rec),
        "F1-Score": float(f1),
        "ROC-AUC": float(auc)
    }


def get_confusion_matrix_data(y_true, y_pred):
    """
    Return confusion matrix raw array and normalized percentages.
    """
    cm = confusion_matrix(y_true, y_pred)
    return cm


def plot_confusion_matrix_fig(cm_or_y_true, y_pred=None, title="Confusion Matrix"):
    """
    Create a clean Seaborn heatmap figure for the confusion matrix.
    Accepts either pre-computed 2x2 confusion matrix array or (y_true, y_pred).
    """
    if y_pred is not None:
        cm = confusion_matrix(cm_or_y_true, y_pred)
    else:
        cm = cm_or_y_true

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        cmap="Blues", 
        cbar=False,
        xticklabels=["Retained (No)", "Churned (Yes)"],
        yticklabels=["Retained (No)", "Churned (Yes)"],
        ax=ax
    )
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Predicted Label", fontsize=10)
    ax.set_ylabel("True Label", fontsize=10)
    plt.tight_layout()
    return fig


def plot_roc_curves_fig(models_dict, X_test, y_test):
    """
    Plot ROC Curves for multiple trained models on the test set.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    
    colors = ["#6366f1", "#ec4899", "#10b981", "#f59e0b", "#8b5cf6"]
    color_idx = 0

    for name, model in models_dict.items():
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc_val = roc_auc_score(y_test, y_prob)
            
            color = colors[color_idx % len(colors)]
            ax.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.3f})", color=color, linewidth=2)
            color_idx += 1

    ax.plot([0, 1], [0, 1], "k--", label="Random Chance (AUC = 0.500)", linewidth=1.5)
    ax.set_xlabel("False Positive Rate", fontsize=10)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=10)
    ax.set_title("Receiver Operating Characteristic (ROC) Curves", fontsize=12, fontweight="bold", pad=10)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    return fig


def plot_feature_importance_fig(importance_series, title="Feature Importance", top_n=15):
    """
    Plot top N native feature importances.
    """
    top_importances = importance_series.sort_values(ascending=False).head(top_n)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        x=top_importances.values, 
        y=top_importances.index, 
        palette="viridis", 
        ax=ax
    )
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Importance Score", fontsize=10)
    ax.set_ylabel("Feature", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5, axis="x")
    plt.tight_layout()
    return fig


def plot_shap_summary_fig(shap_values, X_sample, feature_names, max_display=15):
    """
    Generate SHAP summary plot as a matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    
    # Format feature names cleanly
    clean_names = [f.replace("cat__", "").replace("num__", "") for f in feature_names]
    
    shap.summary_plot(
        shap_values, 
        X_sample, 
        feature_names=clean_names, 
        max_display=max_display, 
        show=False
    )
    plt.title("Global SHAP Feature Summary Plot", fontsize=12, fontweight="bold", pad=15)
    plt.tight_layout()
    return plt.gcf()


def plot_shap_bar_fig(shap_values, feature_names, max_display=15):
    """
    Generate SHAP mean absolute value bar plot.
    """
    clean_names = [f.replace("cat__", "").replace("num__", "") for f in feature_names]
    
    if isinstance(shap_values, np.ndarray):
        mean_shap = np.abs(shap_values).mean(axis=0)
    else:
        mean_shap = np.abs(shap_values.values).mean(axis=0)

    importance_df = pd.Series(mean_shap, index=clean_names).sort_values(ascending=False).head(max_display)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=importance_df.values, y=importance_df.index, palette="mako", ax=ax)
    ax.set_title("SHAP Global Feature Importance (Mean |SHAP Value|)", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Mean |SHAP Value| (Impact on Model Output)", fontsize=10)
    ax.set_ylabel("Feature", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5, axis="x")
    plt.tight_layout()
    return fig


def plot_individual_shap_waterfall_fig(single_shap_exp, max_display=10):
    """
    Generate a SHAP waterfall plot for an individual customer prediction.
    """
    fig = plt.figure(figsize=(8, 5))
    try:
        shap.plots.waterfall(single_shap_exp, max_display=max_display, show=False)
        plt.title("Individual Prediction SHAP Waterfall Plot", fontsize=12, fontweight="bold", pad=15)
        plt.tight_layout()
        return plt.gcf()
    except Exception as e:
        plt.close(fig)
        # Fallback to custom bar chart if waterfall API fails
        fig, ax = plt.subplots(figsize=(8, 5))
        vals = single_shap_exp.values
        names = [f.replace("cat__", "").replace("num__", "") for f in single_shap_exp.feature_names]
        
        df_ind = pd.DataFrame({"Feature": names, "SHAP": vals})
        df_ind["AbsSHAP"] = df_ind["SHAP"].abs()
        df_ind = df_ind.sort_values(by="AbsSHAP", ascending=False).head(max_display).sort_values(by="SHAP")
        
        colors = ["#ef4444" if v > 0 else "#3b82f6" for v in df_ind["SHAP"]]
        ax.barh(df_ind["Feature"], df_ind["SHAP"], color=colors)
        ax.axvline(0, color="black", linestyle="--", alpha=0.7)
        ax.set_title("Customer SHAP Contribution Breakdown", fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("SHAP Value (Positive = Risk Increase, Negative = Risk Decrease)", fontsize=10)
        plt.tight_layout()
        return fig
