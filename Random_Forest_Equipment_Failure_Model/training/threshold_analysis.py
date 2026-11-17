"""
Decision Threshold Optimization & Trade-off Analysis Script
============================================================
Evaluates classification metrics across probability thresholds (0.05 to 0.75)
to determine optimal operational balance between:
- Missed Breakdowns (False Negatives): Severe line shutdown cost
- False Alarms (False Positives): Technician inspection overhead
"""

import sys
from pathlib import Path

training_dir = Path(__file__).resolve().parent
pkg_root = training_dir.parent
src_dir = pkg_root / "src"
for p in [str(src_dir), str(pkg_root)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from feature_engineering import PRIMARY_FEATURES, add_physics_features
import feature_engineering as fe
sys.modules['utils'] = fe

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score, fbeta_score,
    confusion_matrix, balanced_accuracy_score
)

THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]


def run_threshold_analysis():
    print("=" * 80)
    print("DECISION THRESHOLD ANALYSIS - RANDOM FOREST MODEL")
    print("=" * 80)

    # Resolve paths
    model_path = pkg_root / "model" / "ai4i_failure_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at: {model_path}")

    model = joblib.load(model_path)

    # Locate training data
    candidates = [
        pkg_root.parent / "ml" / "data" / "splits",
        pkg_root / "data" / "splits",
        Path("data/splits")
    ]
    splits_dir = next((c for c in candidates if (c / "X_train.csv").exists()), None)
    if splits_dir is None:
        raise FileNotFoundError("Training splits not found in standard paths.")

    X_train = pd.read_csv(splits_dir / "X_train.csv", index_col=0)[PRIMARY_FEATURES]
    y_train = pd.read_csv(splits_dir / "y_train.csv", index_col=0).squeeze()

    # Create 80:20 dev/val split
    _, X_val, _, y_val = train_test_split(
        X_train, y_train,
        test_size=0.20,
        random_state=42,
        stratify=y_train
    )

    y_proba = model.predict_proba(X_val)[:, 1]

    records = []
    for t in THRESHOLDS:
        y_pred = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
        p = precision_score(y_val, y_pred, zero_division=0)
        r = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        f2 = fbeta_score(y_val, y_pred, beta=2, zero_division=0)
        bal_acc = balanced_accuracy_score(y_val, y_pred)

        records.append({
            "Threshold": t,
            "Recall": r,
            "Precision": p,
            "F1-Score": f1,
            "F2-Score": f2,
            "Balanced_Accuracy": bal_acc,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn
        })

    df = pd.DataFrame(records)

    print("\n--- VALIDATION THRESHOLD SWEEP (1,600 SAMPLES) ---")
    formatters = {
        "Threshold": "{:.2f}".format,
        "Recall": lambda x: f"{x*100:.1f}%",
        "Precision": lambda x: f"{x*100:.1f}%",
        "F1-Score": "{:.4f}".format,
        "F2-Score": "{:.4f}".format,
        "Balanced_Accuracy": lambda x: f"{x*100:.1f}%",
        "TP": "{:d}".format,
        "FP": "{:d}".format,
        "FN": "{:d}".format,
        "TN": "{:d}".format
    }
    print(df.to_string(index=False, formatters=formatters))

    best_f2_idx = df["F2-Score"].idxmax()
    best_row = df.loc[best_f2_idx]
    print("\n" + "=" * 80)
    print(f"OPTIMAL F2 THRESHOLD: {best_row['Threshold']:.2f}")
    print(f"  Recall (Failure Detection Rate): {best_row['Recall']*100:.2f}% ({int(best_row['TP'])} caught, {int(best_row['FN'])} missed)")
    print(f"  Precision (Alarm Accuracy):       {best_row['Precision']*100:.2f}% ({int(best_row['FP'])} false alarms)")
    print(f"  F2-Score:                        {best_row['F2-Score']:.4f}")
    print("=" * 80)
    return df


if __name__ == "__main__":
    run_threshold_analysis()
