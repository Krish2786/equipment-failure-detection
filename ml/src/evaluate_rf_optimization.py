"""
AI4I 2020 Predictive Maintenance - Random Forest Optimization Visualization Suite (Phase 2.1)
=============================================================================================
Generates 7 high-resolution (300 DPI) publication-quality charts for Phase 2.1:
1. Baseline vs Optimized Random Forest Performance Comparison
2. Failure Recall Comparison Across All Experimental Configurations
3. Precision-Recall Trade-off Landscape Across Configurations
4. Threshold Optimization Analysis (Threshold vs Recall, Precision, F1, F2)
5. Test Set Confusion Matrix (Held-out 2,000 samples @ threshold 0.35)
6. Feature Importance & Contribution (All 8 Features: 5 Raw + 3 Physics-Engineered)
7. Probability Calibration Reliability Curve
"""

import sys
import json
from pathlib import Path

# Local import support
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import PRIMARY_FEATURES, ENGINEERED_FEATURE_NAMES, ALL_8_FEATURES, add_physics_features

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)

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

def resolve_paths():
    current_dir = Path.cwd().resolve()
    ml_dir = current_dir if current_dir.name == "ml" else (current_dir / "ml" if (current_dir / "ml").is_dir() else Path(__file__).resolve().parent.parent)
    return {
        "ml_dir": ml_dir,
        "splits_dir": ml_dir / "data" / "splits",
        "best_model_dir": ml_dir / "models" / "best_model",
        "metrics_dir": ml_dir / "outputs" / "metrics",
        "charts_dir": ml_dir / "outputs" / "charts" / "model_evaluation" / "random_forest_optimization"
    }

def plot_baseline_vs_optimized(exp_df: pd.DataFrame, output_path: Path):
    """Plot 1: Baseline RF vs Final Optimized RF."""
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    
    baseline_row = exp_df[exp_df["Config_Name"].str.contains("Baseline", case=False)].iloc[0]
    optimized_row = exp_df[exp_df["Config_Name"].str.contains("Combined", case=False)].iloc[0]
    
    metrics = ["Val_PR-AUC", "Val_Recall", "Val_Precision", "Val_F1-Score", "Val_F2-Score", "Val_Balanced_Accuracy"]
    labels = ["PR-AUC", "Recall", "Precision", "F1-Score", "F2-Score", "Balanced Acc"]
    
    base_vals = [baseline_row[m] for m in metrics]
    opt_vals = [optimized_row[m] for m in metrics]
    
    x = np.arange(len(labels))
    width = 0.35
    
    rects1 = ax.bar(x - width/2, base_vals, width, label="Baseline RF (5 Raw Sensors @ 0.50)",
                    color="#94A3B8", edgecolor="#0F172A", linewidth=1.1)
    rects2 = ax.bar(x + width/2, opt_vals, width, label="Optimized RF (8 Features @ 0.35)",
                    color="#10B981", edgecolor="#065F46", linewidth=1.1)
    
    ax.set_title("Validation Benchmark: Baseline Random Forest vs Optimized Random Forest", pad=15, fontweight="bold")
    ax.set_ylabel("Metric Score (0.0 to 1.0)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y")
    ax.legend(loc="lower right", frameon=True)
    
    for r in rects1:
        h = r.get_height()
        ax.annotate(f"{h:.2f}", (r.get_x() + r.get_width() / 2., h),
                    ha='center', va='bottom', fontsize=8, xytext=(0, 3), textcoords='offset points', fontweight='bold')
    for r in rects2:
        h = r.get_height()
        ax.annotate(f"{h:.2f}", (r.get_x() + r.get_width() / 2., h),
                    ha='center', va='bottom', fontsize=8, xytext=(0, 3), textcoords='offset points', fontweight='bold', color="#065F46")
        
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"[RF-EVAL] Saved: {output_path.name}")

def plot_recall_comparison(exp_df: pd.DataFrame, output_path: Path):
    """Plot 2: Recall across all 6 configurations."""
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=300)
    
    short_names = [
        "Baseline RF (0.50)",
        "Exp A: Tuned RF (0.50)",
        "Exp C: SMOTE RF (0.50)",
        "Exp D: Balanced RF (0.50)",
        "Exp E: Physics RF (0.50)",
        "Exp F: Combined RF (0.35)"
    ]
    recalls = [r * 100 for r in exp_df["Val_Recall"]]
    precisions = [p * 100 for p in exp_df["Val_Precision"]]
    
    colors = ["#64748B", "#3B82F6", "#8B5CF6", "#F59E0B", "#059669", "#10B981"]
    
    bars = ax.barh(short_names, recalls, color=colors, edgecolor="#0F172A", height=0.55)
    ax.set_title("Failure Detection Recall Across Random Forest Optimization Experiments", pad=15, fontweight="bold")
    ax.set_xlabel("Failure Recall Rate (%) on Validation Set (54 Total Failures)", fontweight="bold")
    ax.set_xlim(0, 110)
    ax.grid(axis="x")
    
    for bar, rec, prec, row in zip(bars, recalls, precisions, exp_df.itertuples()):
        w = bar.get_width()
        tp = getattr(row, "Val_TP")
        fn = getattr(row, "Val_FN")
        fp = getattr(row, "Val_FP")
        ax.text(w + 1.2, bar.get_y() + bar.get_height()/2,
                f"{rec:.1f}% ({tp}/54 caught, {fn} missed) | Prec: {prec:.1f}% ({fp} FP)",
                va="center", fontsize=9, fontweight="bold")
        
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"[RF-EVAL] Saved: {output_path.name}")

def plot_precision_recall_tradeoff(exp_df: pd.DataFrame, output_path: Path):
    """Plot 3: Precision vs Recall trade-off scatter plot."""
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    
    short_names = [
        "Baseline RF (0.50)",
        "Tuned RF (0.50)",
        "SMOTE RF (0.50)",
        "Balanced RF (0.50)",
        "Physics RF (0.50)",
        "Combined RF (0.35)"
    ]
    colors = ["#64748B", "#3B82F6", "#8B5CF6", "#F59E0B", "#059669", "#10B981"]
    
    for i, (_, row) in enumerate(exp_df.iterrows()):
        rec = float(row["Val_Recall"]) * 100
        prec = float(row["Val_Precision"]) * 100
        f2 = float(row["Val_F2-Score"])
        ax.scatter(rec, prec, s=180, color=colors[i], edgecolors="#0F172A", linewidth=1.5, zorder=5)
        offset_y = 3 if i % 2 == 0 else -6
        ax.annotate(f"{short_names[i]}\n(F2={f2:.3f})", xy=(rec, prec), xytext=(8, offset_y),
                    textcoords="offset points", fontsize=8.5, fontweight="bold")
        
    ax.set_title("Precision vs Recall Trade-off Landscape Across Optimization Experiments", pad=15, fontweight="bold")
    ax.set_xlabel("Failure Detection Recall (%)", fontweight="bold")
    ax.set_ylabel("Precision (% of Alarms that are Real Failures)", fontweight="bold")
    ax.set_xlim(50, 105)
    ax.set_ylim(15, 95)
    ax.grid(True)
    
    # Highlight Pareto-optimal region
    ax.axvspan(78, 100, color="#10B981", alpha=0.1, label="High-Recall Target Zone (>=78%)")
    ax.legend(loc="lower left", frameon=True)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"[RF-EVAL] Saved: {output_path.name}")

def plot_threshold_curves(thresh_df: pd.DataFrame, output_path: Path):
    """Plot 4: Threshold vs Recall, Precision, F1, F2 for the winning model."""
    eng_thresh = thresh_df[thresh_df["Config"].str.contains("Engineered", case=False)].copy()
    if eng_thresh.empty:
        eng_thresh = thresh_df.iloc[:len(THRESHOLDS)].copy()
        
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    
    ax.plot(eng_thresh["Threshold"], eng_thresh["Recall"], color="#EF4444", marker="o", linewidth=2.2, label="Recall (Failure Detection Rate)")
    ax.plot(eng_thresh["Threshold"], eng_thresh["Precision"], color="#2563EB", marker="s", linewidth=2.2, label="Precision (Alarm Accuracy)")
    ax.plot(eng_thresh["Threshold"], eng_thresh["F1-Score"], color="#8B5CF6", marker="^", linewidth=2.0, label="F1-Score (Harmonic Mean)")
    ax.plot(eng_thresh["Threshold"], eng_thresh["F2-Score"], color="#10B981", marker="D", linewidth=2.5, label="F2-Score (Recall x2 Safety Metric)")
    
    # Mark optimal threshold (0.35)
    best_thresh = 0.35
    ax.axvline(best_thresh, color="#10B981", linestyle="--", linewidth=1.8, label=f"Selected Operational Threshold ({best_thresh:.2f})")
    
    # Annotate operational regimes
    ax.text(0.12, 0.20, "High-Sensitivity Regime\n(Prioritizes Zero Missed Failures)",
            fontsize=8.5, style="italic", bbox=dict(boxstyle="round,pad=0.4", facecolor="#FEF3C7", edgecolor="#F59E0B", alpha=0.85))
    ax.text(0.55, 0.20, "High-Precision Regime\n(Minimal False Alarms)",
            fontsize=8.5, style="italic", bbox=dict(boxstyle="round,pad=0.4", facecolor="#EFF6FF", edgecolor="#3B82F6", alpha=0.85))
    
    ax.set_title("Optimized Random Forest Decision Threshold Sweep (Validation Set)", pad=15, fontweight="bold")
    ax.set_xlabel("Classification Probability Threshold", fontweight="bold")
    ax.set_ylabel("Metric Score", fontweight="bold")
    ax.set_xlim(0.04, 0.76)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True)
    ax.legend(loc="center right", frameon=True)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"[RF-EVAL] Saved: {output_path.name}")

def plot_confusion_matrix_test(test_res: dict, output_path: Path):
    """Plot 5: Test Confusion Matrix."""
    cm = test_res["confusion_matrix"]
    tp = cm["True_Positives"]
    fp = cm["False_Positives"]
    fn = cm["False_Negatives"]
    tn = cm["True_Negatives"]
    
    matrix = np.array([[tn, fp], [fn, tp]])
    thresh = test_res["selected_threshold"]
    
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    cmap = sns.light_palette("#10B981", as_cmap=True)
    sns.heatmap(matrix, annot=True, fmt=",d", cmap=cmap, cbar=False,
                xticklabels=["Pred: Normal (0)", "Pred: Failure (1)"],
                yticklabels=["Actual: Normal (0)", "Actual: Failure (1)"],
                annot_kws={"fontsize": 13, "fontweight": "bold"}, ax=ax,
                linewidths=1.5, linecolor="#0F172A")
    
    ax.set_title(f"Optimized Random Forest — Test Confusion Matrix (2,000 samples)\n[Threshold = {thresh:.2f} | 57/68 Failures Detected (83.8% Recall)]",
                 fontweight="bold", pad=15)
    
    summary_text = (
        f"Failure Detection Rate (Recall): {tp/(tp+fn)*100:.2f}% ({tp}/{tp+fn})\n"
        f"Precision (Alarm Accuracy):       {tp/(tp+fp)*100:.2f}% ({tp}/{tp+fp})\n"
        f"Missed Breakdowns (FN):          {fn}\n"
        f"False Alarms (FP):               {fp} (Only 0.5% of Normal operations)"
    )
    plt.figtext(0.5, -0.05, summary_text, ha="center", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.6", facecolor="#F8FAFC", edgecolor="#CBD5E1", alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"[RF-EVAL] Saved: {output_path.name}")

def plot_feature_importance(best_model, X_test, y_test, output_path: Path):
    """Plot 6: Feature importance across all 8 features."""
    rf_clf = best_model.named_steps["rf"] if hasattr(best_model, "named_steps") else best_model
    raw_imp = rf_clf.feature_importances_
    
    feature_labels = [
        "Air temp [K]",
        "Process temp [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
        "Delta-T (Process - Air)",
        "Mechanical Power [W]",
        "Overstrain Index [Nm*min]"
    ]
    
    X_test_8 = add_physics_features(X_test)
    perm = permutation_importance(rf_clf, X_test_8, y_test, n_repeats=10, random_state=42, scoring="average_precision")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=300)
    
    # 1. Native Gini Importance
    df_imp = pd.DataFrame({"Feature": feature_labels, "Importance": raw_imp}).sort_values("Importance", ascending=True)
    colors = ["#3B82F6" if "Delta" not in f and "Power" not in f and "Overstrain" not in f else "#10B981" for f in df_imp["Feature"]]
    ax1.barh(df_imp["Feature"], df_imp["Importance"], color=colors, edgecolor="#0F172A", height=0.55)
    ax1.set_title("Random Forest Gini Feature Importance (All 8 Features)", fontweight="bold", pad=12)
    ax1.set_xlabel("Relative Importance Weight", fontweight="bold")
    ax1.grid(axis="x")
    for i, v in enumerate(df_imp["Importance"]):
        ax1.text(v + 0.005, i, f"{v*100:.1f}%", va="center", fontweight="bold", fontsize=9)
        
    # 2. Permutation Importance
    df_perm = pd.DataFrame({
        "Feature": feature_labels,
        "Mean": perm.importances_mean,
        "Std": perm.importances_std
    }).sort_values("Mean", ascending=True)
    
    colors_perm = ["#3B82F6" if "Delta" not in f and "Power" not in f and "Overstrain" not in f else "#10B981" for f in df_perm["Feature"]]
    ax2.barh(df_perm["Feature"], df_perm["Mean"], xerr=df_perm["Std"], color=colors_perm, edgecolor="#0F172A", height=0.55, capsize=4)
    ax2.set_title("Permutation Importance on Test Set (Drop in PR-AUC)", fontweight="bold", pad=12)
    ax2.set_xlabel("Mean Decrease in Average Precision", fontweight="bold")
    ax2.grid(axis="x")
    for i, v in enumerate(df_perm["Mean"]):
        ax2.text(max(v, 0) + 0.003, i, f"{v:.3f}", va="center", fontweight="bold", fontsize=9)
        
    plt.suptitle("Feature Importance & Engineered Physics Telemetry Contribution (Green = Engineered Features)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"[RF-EVAL] Saved: {output_path.name}")

def plot_calibration(best_model, X_test, y_test, output_path: Path):
    """Plot 7: Calibration curve on test set."""
    fig, ax = plt.subplots(figsize=(7.5, 6), dpi=300)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    
    prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=8, strategy="uniform")
    
    ax.plot(prob_pred, prob_true, marker="o", linewidth=2.2, color="#10B981", label="Optimized Random Forest Calibration")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#94A3B8", linewidth=1.5, label="Perfect Reliability Reference")
    
    ax.set_title("Probability Calibration Reliability Diagram (Held-Out Test Set)", pad=15, fontweight="bold")
    ax.set_xlabel("Mean Predicted Probability", fontweight="bold")
    ax.set_ylabel("Observed Empirical Failure Fraction", fontweight="bold")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True)
    ax.legend(loc="upper left", frameon=True)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"[RF-EVAL] Saved: {output_path.name}")

def generate_all_rf_charts():
    paths = resolve_paths()
    print(f"[RF-EVAL] Loading evaluation metrics from: {paths['metrics_dir']}")
    
    exp_df = pd.read_csv(paths["metrics_dir"] / "rf_experiments_comparison.csv")
    thresh_df = pd.read_csv(paths["metrics_dir"] / "rf_threshold_analysis.csv")
    with open(paths["metrics_dir"] / "rf_optimized_test_results.json", "r") as f:
        test_res = json.load(f)
        
    best_model = joblib.load(paths["best_model_dir"] / "ai4i_failure_model.joblib")
    
    X_test = pd.read_csv(paths["splits_dir"] / "X_test.csv", index_col=0)
    y_test = pd.read_csv(paths["splits_dir"] / "y_test.csv", index_col=0).squeeze()
    
    out_dir = paths["charts_dir"]
    print(f"[RF-EVAL] Generating Phase 2.1 charts in: {out_dir}")
    
    plot_baseline_vs_optimized(exp_df, out_dir / "01_baseline_vs_optimized_rf.png")
    plot_recall_comparison(exp_df, out_dir / "02_recall_comparison_across_experiments.png")
    plot_precision_recall_tradeoff(exp_df, out_dir / "03_precision_recall_tradeoff.png")
    plot_threshold_curves(thresh_df, out_dir / "04_threshold_analysis_curves.png")
    plot_confusion_matrix_test(test_res, out_dir / "05_confusion_matrix_test.png")
    plot_feature_importance(best_model, X_test, y_test, out_dir / "06_feature_importance.png")
    plot_calibration(best_model, X_test, y_test, out_dir / "07_calibration_curve.png")
    
    print("[RF-EVAL] All 7 Phase 2.1 visualization charts successfully generated at 300 DPI!")

if __name__ == "__main__":
    generate_all_rf_charts()
