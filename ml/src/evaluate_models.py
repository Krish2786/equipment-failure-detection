"""
AI4I 2020 Predictive Maintenance - Model Evaluation & Visualization Suite
=========================================================================
Generates 10 high-resolution (300 DPI) publication-quality evaluation charts:
1. Model Comparison Chart (Validation partition)
2. 5-Fold Cross-Validation Performance Comparison
3. Confusion Matrix of the Selected Model (Held-out Test Set)
4. Receiver Operating Characteristic (ROC) Curves Comparison
5. Precision-Recall (PR) Curves Comparison
6. Feature Importance & Permutation Importance Analysis
7. Threshold vs Precision & Recall Curves
8. Threshold vs F1-Score & F2-Score Trade-off Curves
9. Probability Calibration Curve (Reliability Diagram)
10. Train vs Validation Performance Comparison (Overfitting Diagnostic)
"""

import sys
import json
from pathlib import Path

from typing import Any, Dict, List
# Local import support
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import convert_to_numpy, PRIMARY_FEATURES, TARGET

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    fbeta_score
)

# High-resolution styling
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "axes.edgecolor": "#CBD5E1",
    "axes.linewidth": 1.2,
    "grid.color": "#E2E8F0",
    "grid.linestyle": "--",
    "grid.alpha": 0.7,
    "xtick.color": "#334155",
    "ytick.color": "#334155",
    "axes.labelcolor": "#1E293B",
    "figure.titlesize": 14,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10
})

PALETTE = {
    "xgboost": "#2563EB",       # Royal Blue
    "random_forest": "#059669", # Emerald
    "decision_tree": "#D97706", # Amber
    "dummy": "#94A3B8",         # Slate Gray
    "normal": "#10B981",
    "failure": "#EF4444"
}

def resolve_paths():
    current_dir = Path.cwd().resolve()
    ml_dir = current_dir if current_dir.name == "ml" else (current_dir / "ml" if (current_dir / "ml").is_dir() else Path(__file__).resolve().parent.parent)
    return {
        "ml_dir": ml_dir,
        "splits_dir": ml_dir / "data" / "splits",
        "candidates_dir": ml_dir / "models" / "candidates",
        "best_model_dir": ml_dir / "models" / "best_model",
        "metrics_dir": ml_dir / "outputs" / "metrics",
        "charts_dir": ml_dir / "outputs" / "charts" / "model_evaluation"
    }

def plot_model_comparison(val_df: pd.DataFrame, output_path: Path):
    """Plot 1: Model comparison on validation set."""
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    plot_df = val_df[val_df["Model"] != "Dummy Classifier"].copy()
    metrics = ["PR-AUC", "Recall", "Precision", "F1-Score", "ROC-AUC"]
    
    melted = pd.melt(plot_df, id_vars=["Model"], value_vars=metrics, var_name="Metric", value_name="Score")
    
    sns.barplot(data=melted, x="Metric", y="Score", hue="Model",
                palette=[PALETTE["decision_tree"], PALETTE["random_forest"], PALETTE["xgboost"]],
                edgecolor="#0F172A", linewidth=1.1, ax=ax)
    
    ax.set_title("Validation Performance Comparison Across Machine Learning Candidates", pad=15, fontweight="bold")
    ax.set_ylabel("Metric Score (0.0 to 1.0)", fontweight="bold")
    ax.set_xlabel("Evaluation Metric", fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y")
    ax.legend(title="Candidate Model", loc="lower right", frameon=True)
    
    # Annotate bars
    for p in ax.patches:
        h = p.get_height()
        if h > 0.05:
            ax.annotate(f"{h:.2f}", (p.get_x() + p.get_width() / 2., h),
                        ha='center', va='bottom', fontsize=8, xytext=(0, 3),
                        textcoords='offset points', fontweight='bold')
            
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"[EVAL] Saved: {output_path.name}")

def plot_cv_performance(cv_df: pd.DataFrame, output_path: Path):
    """Plot 2: Cross-validation performance with fold variance."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    plot_cv = cv_df[cv_df["Model"] != "Dummy Classifier"].copy()
    
    # PR-AUC fold distribution
    sns.boxplot(data=plot_cv, x="Model", y="PR-AUC", ax=ax1,
                palette=[PALETTE["decision_tree"], PALETTE["random_forest"], PALETTE["xgboost"]],
                boxprops=dict(alpha=0.8), fliersize=4)
    sns.stripplot(data=plot_cv, x="Model", y="PR-AUC", ax=ax1, color="#0F172A", size=6, jitter=0.2)
    ax1.set_title("5-Fold Cross-Validation: PR-AUC (Average Precision)", fontweight="bold", pad=12)
    ax1.set_ylabel("PR-AUC Score", fontweight="bold")
    ax1.grid(axis="y")
    
    # Recall vs Precision fold comparison
    sns.boxplot(data=plot_cv, x="Model", y="Recall", ax=ax2,
                palette=[PALETTE["decision_tree"], PALETTE["random_forest"], PALETTE["xgboost"]],
                boxprops=dict(alpha=0.8), fliersize=4)
    sns.stripplot(data=plot_cv, x="Model", y="Recall", ax=ax2, color="#0F172A", size=6, jitter=0.2)
    ax2.set_title("5-Fold Cross-Validation: Failure Class Recall", fontweight="bold", pad=12)
    ax2.set_ylabel("Recall Score", fontweight="bold")
    ax2.grid(axis="y")
    
    plt.suptitle("5-Fold Stratified Cross-Validation Stability on Dev-Train Partition (6,400 samples)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"[EVAL] Saved: {output_path.name}")

def plot_confusion_matrix(test_results: dict, output_path: Path):
    """Plot 3: Confusion matrix of final model on held-out test set."""
    cm = test_results["confusion_matrix"]
    tp = cm["True Positives (Detected Failures)"]
    fp = cm["False Positives (False Alarms)"]
    fn = cm["False Negatives (Missed Failures)"]
    tn = cm["True Negatives (Correct Normal)"]
    
    matrix = np.array([[tn, fp], [fn, tp]])
    model_name = test_results["model_name"]
    thresh = test_results["selected_threshold"]
    
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    cmap = sns.light_palette("#2563EB", as_cmap=True)
    sns.heatmap(matrix, annot=True, fmt=",d", cmap=cmap, cbar=False,
                xticklabels=["Pred: Normal (0)", "Pred: Failure (1)"],
                yticklabels=["Actual: Normal (0)", "Actual: Failure (1)"],
                annot_kws={"fontsize": 13, "fontweight": "bold"}, ax=ax,
                linewidths=1.5, linecolor="#0F172A")
    
    ax.set_title(f"Confusion Matrix on Held-Out Test Set (2,000 samples)\n[{model_name} @ Decision Threshold {thresh:.2f}]",
                 fontweight="bold", pad=15)
    
    total_failures = tp + fn
    recall_pct = (tp / total_failures) * 100
    precision_pct = (tp / (tp + fp)) * 100
    
    summary_text = (
        f"Failure Detection Rate (Recall): {recall_pct:.1f}% ({tp}/{total_failures})\n"
        f"Precision (Alarm Accuracy):       {precision_pct:.1f}%\n"
        f"Missed Breakdowns (FN):          {fn}\n"
        f"False Alarms (FP):               {fp}"
    )
    plt.figtext(0.5, -0.05, summary_text, ha="center", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.6", facecolor="#F8FAFC", edgecolor="#CBD5E1", alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"[EVAL] Saved: {output_path.name}")

def plot_roc_curves(fitted_models: dict, X_val: pd.DataFrame, y_val: pd.Series, output_path: Path):
    """Plot 4: ROC Curves Comparison."""
    fig, ax = plt.subplots(figsize=(8, 6.5), dpi=300)
    colors = {
        "Random Forest": PALETTE["random_forest"],
        "XGBoost": PALETTE["xgboost"],
        "Decision Tree": PALETTE["decision_tree"],
        "Dummy Classifier": PALETTE["dummy"]
    }
    
    for name, model in fitted_models.items():
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_val)[:, 1]
            fpr, tpr, _ = roc_curve(y_val, y_proba)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=colors.get(name, "#000"), linewidth=2.2,
                    label=f"{name} (AUC = {roc_auc:.4f})")
            
    ax.plot([0, 1], [0, 1], color="#94A3B8", linestyle="--", linewidth=1.5, label="Random Chance Baseline (AUC = 0.5000)")
    ax.set_title("Receiver Operating Characteristic (ROC) Curves - Validation Set", pad=15, fontweight="bold")
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontweight="bold")
    ax.set_ylabel("True Positive Rate (Recall)", fontweight="bold")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True)
    ax.legend(loc="lower right", frameon=True, facecolor="#F8FAFC", edgecolor="#CBD5E1")
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"[EVAL] Saved: {output_path.name}")

def plot_pr_curves(fitted_models: dict, X_val: pd.DataFrame, y_val: pd.Series, output_path: Path):
    """Plot 5: Precision-Recall Curves Comparison."""
    fig, ax = plt.subplots(figsize=(8, 6.5), dpi=300)
    colors = {
        "Random Forest": PALETTE["random_forest"],
        "XGBoost": PALETTE["xgboost"],
        "Decision Tree": PALETTE["decision_tree"],
        "Dummy Classifier": PALETTE["dummy"]
    }
    
    base_rate = y_val.sum() / len(y_val)
    ax.axhline(base_rate, color="#94A3B8", linestyle="--", linewidth=1.5,
               label=f"No-Skill Baseline ({base_rate*100:.2f}%)")
    
    for name, model in fitted_models.items():
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_val)[:, 1]
            precision, recall, _ = precision_recall_curve(y_val, y_proba)
            ap = average_precision_score(y_val, y_proba)
            ax.plot(recall, precision, color=colors.get(name, "#000"), linewidth=2.2,
                    label=f"{name} (PR-AUC = {ap:.4f})")
            
    ax.set_title("Precision-Recall (PR) Curves - Validation Set (Imbalanced Benchmark)", pad=15, fontweight="bold")
    ax.set_xlabel("Recall (Failure Detection Rate)", fontweight="bold")
    ax.set_ylabel("Precision (Positive Predictive Value)", fontweight="bold")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True)
    ax.legend(loc="upper right", frameon=True, facecolor="#F8FAFC", edgecolor="#CBD5E1")
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"[EVAL] Saved: {output_path.name}")

def plot_feature_importance(best_model: Any, X_test: pd.DataFrame, y_test: pd.Series, output_path: Path):
    """Plot 6: Built-in feature importance and permutation importance."""
    # Extract underlying classifier if pipeline
    clf = best_model.named_steps["classifier"] if hasattr(best_model, "named_steps") else best_model
    
    # 1. Gini / Gain importance
    if hasattr(clf, "feature_importances_"):
        raw_imp = clf.feature_importances_
    else:
        raw_imp = np.ones(len(PRIMARY_FEATURES)) / len(PRIMARY_FEATURES)
        
    # 2. Permutation importance on test set
    perm = permutation_importance(best_model, X_test, y_test, n_repeats=10, random_state=42, scoring="average_precision")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5), dpi=300)
    
    # Plot 1: Model Feature Importance
    feat_df = pd.DataFrame({"Feature": PRIMARY_FEATURES, "Importance": raw_imp}).sort_values("Importance", ascending=True)
    ax1.barh(feat_df["Feature"], feat_df["Importance"], color="#3B82F6", edgecolor="#1E3A8A", height=0.55)
    ax1.set_title("Native Model Feature Importance (Gini / Gain)", fontweight="bold", pad=12)
    ax1.set_xlabel("Normalized Relative Importance", fontweight="bold")
    ax1.grid(axis="x")
    for i, v in enumerate(feat_df["Importance"]):
        ax1.text(v + 0.01, i, f"{v*100:.1f}%", va="center", fontweight="bold", fontsize=9)
        
    # Plot 2: Permutation Importance
    perm_df = pd.DataFrame({
        "Feature": PRIMARY_FEATURES,
        "Mean": perm.importances_mean,
        "Std": perm.importances_std
    }).sort_values("Mean", ascending=True)
    
    ax2.barh(perm_df["Feature"], perm_df["Mean"], xerr=perm_df["Std"],
             color="#10B981", edgecolor="#065F46", height=0.55, capsize=4)
    ax2.set_title("Permutation Importance on Test Set (PR-AUC Drop)", fontweight="bold", pad=12)
    ax2.set_xlabel("Mean PR-AUC Metric Decrease upon Permutation", fontweight="bold")
    ax2.grid(axis="x")
    for i, v in enumerate(perm_df["Mean"]):
        ax2.text(v + 0.005, i, f"{v:.3f}", va="center", fontweight="bold", fontsize=9)
        
    plt.suptitle("Feature Importance & Predictive Telemetry Contribution Analysis", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"[EVAL] Saved: {output_path.name}")

def plot_threshold_vs_pr(tuning_df: pd.DataFrame, output_path: Path):
    """Plot 7: Threshold vs Precision and Recall."""
    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=300)
    
    ax.plot(tuning_df["Threshold"], tuning_df["Recall"], color="#EF4444", marker="o", linewidth=2.2, label="Recall (Failure Capture Rate)")
    ax.plot(tuning_df["Threshold"], tuning_df["Precision"], color="#2563EB", marker="s", linewidth=2.2, label="Precision (True Alarm Proportion)")
    
    ax.set_title("Decision Threshold vs Precision & Recall (Validation Set Trade-off)", pad=15, fontweight="bold")
    ax.set_xlabel("Classification Probability Threshold", fontweight="bold")
    ax.set_ylabel("Score", fontweight="bold")
    ax.set_xlim(0.05, 0.80)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True)
    ax.legend(loc="center right", frameon=True, facecolor="#F8FAFC", edgecolor="#CBD5E1")
    
    # Highlight cross-over / default 0.50
    ax.axvline(0.50, color="#64748B", linestyle="--", linewidth=1.5, label="Default Threshold (0.50)")
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"[EVAL] Saved: {output_path.name}")

def plot_threshold_vs_f_scores(tuning_df: pd.DataFrame, output_path: Path):
    """Plot 8: Threshold vs F1 and F2 scores."""
    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=300)
    
    ax.plot(tuning_df["Threshold"], tuning_df["F1-Score"], color="#8B5CF6", marker="^", linewidth=2.2, label="F1-Score (Harmonic Mean)")
    ax.plot(tuning_df["Threshold"], tuning_df["F2-Score"], color="#059669", marker="D", linewidth=2.2, label="F2-Score (Recall Weighted x2)")
    
    best_f2_thresh = tuning_df.loc[tuning_df["F2-Score"].idxmax(), "Threshold"]
    max_f2 = tuning_df["F2-Score"].max()
    
    ax.axvline(best_f2_thresh, color="#059669", linestyle="-.", linewidth=1.8,
               label=f"Optimal F2 Threshold ({best_f2_thresh:.2f})")
    ax.scatter([best_f2_thresh], [max_f2], color="#059669", s=100, zorder=5)
    
    ax.set_title("Decision Threshold vs F1 & F2 Metrics (Maintenance Cost Optimization)", pad=15, fontweight="bold")
    ax.set_xlabel("Probability Threshold", fontweight="bold")
    ax.set_ylabel("Score", fontweight="bold")
    ax.set_xlim(0.05, 0.80)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True)
    ax.legend(loc="lower center", frameon=True, facecolor="#F8FAFC", edgecolor="#CBD5E1")
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"[EVAL] Saved: {output_path.name}")

def plot_calibration_curve(best_model: Any, X_val: pd.DataFrame, y_val: pd.Series, output_path: Path):
    """Plot 9: Calibration reliability curve."""
    fig, ax = plt.subplots(figsize=(7.5, 6), dpi=300)
    y_proba = best_model.predict_proba(X_val)[:, 1]
    
    prob_true, prob_pred = calibration_curve(y_val, y_proba, n_bins=8, strategy="uniform")
    
    ax.plot(prob_pred, prob_true, marker="o", linewidth=2.2, color="#2563EB", label="Selected Model Probability Calibration")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#94A3B8", linewidth=1.5, label="Perfectly Calibrated Reference")
    
    ax.set_title("Probability Calibration Reliability Diagram (Validation Set)", pad=15, fontweight="bold")
    ax.set_xlabel("Mean Predicted Probability", fontweight="bold")
    ax.set_ylabel("Observed Empirical Failure Fraction", fontweight="bold")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True)
    ax.legend(loc="upper left", frameon=True)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"[EVAL] Saved: {output_path.name}")

def plot_train_vs_val(cv_df: pd.DataFrame, val_df: pd.DataFrame, output_path: Path):
    """Plot 10: Training vs Validation performance."""
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    models = ["Decision Tree", "Random Forest", "XGBoost"]
    
    comp_data = []
    for m in models:
        cv_pr = cv_df[cv_df["Model"] == m]["PR-AUC"].mean()
        val_pr = val_df[val_df["Model"] == m]["PR-AUC"].values[0]
        comp_data.append({"Model": m, "Partition": "5-Fold CV Mean", "PR-AUC": cv_pr})
        comp_data.append({"Model": m, "Partition": "Holdout Validation", "PR-AUC": val_pr})
        
    df_plot = pd.DataFrame(comp_data)
    sns.barplot(data=df_plot, x="Model", y="PR-AUC", hue="Partition",
                palette=["#93C5FD", "#1D4ED8"], edgecolor="#0F172A", linewidth=1.1, ax=ax)
    
    ax.set_title("Cross-Validation vs Holdout Validation Generalization (PR-AUC)", pad=15, fontweight="bold")
    ax.set_ylabel("PR-AUC Score", fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y")
    ax.legend(loc="lower right", frameon=True)
    
    for p in ax.patches:
        h = p.get_height()
        if h > 0.05:
            ax.annotate(f"{h:.3f}", (p.get_x() + p.get_width() / 2., h),
                        ha='center', va='bottom', fontsize=9, xytext=(0, 3),
                        textcoords='offset points', fontweight='bold')
            
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"[EVAL] Saved: {output_path.name}")

def generate_all_evaluation_charts():
    paths = resolve_paths()
    print(f"[EVAL] Loading evaluation artifacts from: {paths['metrics_dir']}")
    
    val_df = pd.read_csv(paths["metrics_dir"] / "validation_results.csv")
    cv_df = pd.read_csv(paths["metrics_dir"] / "cross_validation_results.csv")
    with open(paths["metrics_dir"] / "test_results.json", "r") as f:
        test_results = json.load(f)
        
    tuning_df = pd.read_csv(paths["metrics_dir"] / "threshold_tuning_validation.csv")
    
    # Load splits for curve calculation
    X_train = pd.read_csv(paths["splits_dir"] / "X_train.csv", index_col=0)
    y_train = pd.read_csv(paths["splits_dir"] / "y_train.csv", index_col=0).squeeze()
    X_test = pd.read_csv(paths["splits_dir"] / "X_test.csv", index_col=0)
    y_test = pd.read_csv(paths["splits_dir"] / "y_test.csv", index_col=0).squeeze()
    
    _, X_val, _, y_val = train_test_split(
        X_train, y_train, test_size=0.20, random_state=42, stratify=y_train
    )
    
    # Load candidate models
    candidates = {}
    for mod_name, file_name in [
        ("Dummy Classifier", "dummy_classifier_model.joblib"),
        ("Decision Tree", "decision_tree_model.joblib"),
        ("Random Forest", "random_forest_model.joblib"),
        ("XGBoost", "xgboost_model.joblib")
    ]:
        p = paths["candidates_dir"] / file_name
        if p.exists():
            candidates[mod_name] = joblib.load(p)
            
    best_model = joblib.load(paths["best_model_dir"] / "ai4i_failure_model.joblib")
    out_dir = paths["charts_dir"]
    
    print(f"[EVAL] Generating 10 publication charts in: {out_dir}")
    plot_model_comparison(val_df, out_dir / "01_model_comparison_validation.png")
    plot_cv_performance(cv_df, out_dir / "02_cross_validation_performance.png")
    plot_confusion_matrix(test_results, out_dir / "03_confusion_matrix_test.png")
    plot_roc_curves(candidates, X_val, y_val, out_dir / "04_roc_curve_comparison.png")
    plot_pr_curves(candidates, X_val, y_val, out_dir / "05_precision_recall_curve.png")
    plot_feature_importance(best_model, X_test, y_test, out_dir / "06_feature_importance.png")
    plot_threshold_vs_pr(tuning_df, out_dir / "07_threshold_vs_precision_recall.png")
    plot_threshold_vs_f_scores(tuning_df, out_dir / "08_threshold_vs_f1_f2.png")
    plot_calibration_curve(best_model, X_val, y_val, out_dir / "09_calibration_curve.png")
    plot_train_vs_val(cv_df, val_df, out_dir / "10_train_vs_validation_performance.png")
    
    print("[EVAL] All 10 evaluation charts successfully generated at 300 DPI!")

if __name__ == "__main__":
    generate_all_evaluation_charts()
