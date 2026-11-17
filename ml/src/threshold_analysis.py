"""
AI4I 2020 Predictive Maintenance - Decision Threshold Analysis
==============================================================
Evaluates classification performance across multiple probability thresholds
on the validation partition.

Analyzes the industrial trade-off between:
- Missed Breakdowns (False Negatives): Unplanned line shutdown, catastrophic damage.
- False Maintenance Alarms (False Positives): Technician dispatch and inspection cost.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, List

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import convert_to_numpy, PRIMARY_FEATURES, TARGET

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    confusion_matrix
)

THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]

def evaluate_threshold_sweep(y_val: np.ndarray, y_proba: np.ndarray, model_name: str) -> pd.DataFrame:
    """Evaluates metrics across the threshold grid."""
    records = []
    for t in THRESHOLDS:
        y_pred = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
        p = precision_score(y_val, y_pred, zero_division=0)
        r = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        f2 = fbeta_score(y_val, y_pred, beta=2, zero_division=0)

        records.append({
            "Model": model_name,
            "Threshold": t,
            "Precision": float(p),
            "Recall": float(r),
            "F1-Score": float(f1),
            "F2-Score": float(f2),
            "TP": int(tp),
            "FP": int(fp),
            "FN": int(fn),
            "TN": int(tn),
            "False_Alarm_Rate": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        })
    return pd.DataFrame(records)

def run_threshold_analysis():
    current_dir = Path.cwd().resolve()
    ml_dir = current_dir if current_dir.name == "ml" else (current_dir / "ml" if (current_dir / "ml").is_dir() else Path(__file__).resolve().parent.parent)

    splits_dir = ml_dir / "data" / "splits"
    candidates_dir = ml_dir / "models" / "candidates"
    metrics_dir = ml_dir / "outputs" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    X_train = pd.read_csv(splits_dir / "X_train.csv", index_col=0)
    y_train = pd.read_csv(splits_dir / "y_train.csv", index_col=0).squeeze()

    X_dev, X_val, y_dev, y_val = train_test_split(
        X_train, y_train, test_size=0.20, random_state=42, stratify=y_train
    )

    models_to_check = {
        "Random Forest": candidates_dir / "random_forest_model.joblib",
        "XGBoost": candidates_dir / "xgboost_model.joblib",
        "Decision Tree": candidates_dir / "decision_tree_model.joblib"
    }

    all_dfs = []
    print("=" * 80)
    print("DECISION THRESHOLD ANALYSIS ON VALIDATION SET (1,600 SAMPLES)")
    print("=" * 80)

    for name, path in models_to_check.items():
        if not path.exists():
            continue
        model = joblib.load(path)
        if not hasattr(model, "predict_proba"):
            continue

        y_proba = model.predict_proba(X_val)[:, 1]
        df_sweep = evaluate_threshold_sweep(y_val.values, y_proba, name)
        all_dfs.append(df_sweep)

        best_f2_idx = df_sweep["F2-Score"].idxmax()
        best_f2_row = df_sweep.loc[best_f2_idx]
        print(f"\n--- Model: {name} ---")
        print(f"Optimal F2 Threshold: {best_f2_row['Threshold']:.2f}")
        print(f"  Recall (Failures Caught): {best_f2_row['Recall']*100:.1f}% ({int(best_f2_row['TP'])} / {int(best_f2_row['TP'] + best_f2_row['FN'])})")
        print(f"  Precision (True Alarms):  {best_f2_row['Precision']*100:.1f}%")
        print(f"  F1-Score:                 {best_f2_row['F1-Score']:.4f}")
        print(f"  F2-Score:                 {best_f2_row['F2-Score']:.4f}")
        print(f"  False Alarms (FP):        {int(best_f2_row['FP'])}")
        print(f"  Missed Breakdowns (FN):   {int(best_f2_row['FN'])}")

    combined_df = pd.concat(all_dfs, ignore_index=True)
    out_csv = metrics_dir / "threshold_comparison_all_models.csv"
    combined_df.to_csv(out_csv, index=False)
    print(f"\nSaved combined threshold comparison to: {out_csv}")
    print("=" * 80)

if __name__ == "__main__":
    run_threshold_analysis()
