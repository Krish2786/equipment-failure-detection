"""
Equipment Failure Model Evaluation Script
=========================================
Evaluates the serialized Random Forest model on test telemetry data.
Calculates industrial predictive maintenance metrics:
- Recall (Failure Detection Rate)
- Precision (Alarm Reliability)
- F1 and F2 (Recall-Weighted) Scores
- PR-AUC and ROC-AUC
- Balanced Accuracy and Brier Calibration Score
- Confusion Matrix
"""

import sys
import json
from pathlib import Path
from typing import Optional

# Setup path and register unpickling alias
src_dir = Path(__file__).resolve().parent
pkg_root = src_dir.parent
for p in [str(src_dir), str(pkg_root)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from .feature_engineering import PRIMARY_FEATURES, TARGET
    from . import feature_engineering as fe
except ImportError:
    from feature_engineering import PRIMARY_FEATURES, TARGET
    import feature_engineering as fe

sys.modules['utils'] = fe

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    classification_report
)


def evaluate_dataset(
    model_path: Optional[Path] = None,
    test_features_csv: Optional[Path] = None,
    test_labels_csv: Optional[Path] = None,
    threshold: float = 0.35
):
    """
    Evaluates the model on test features and labels.
    """
    if model_path is None:
        model_path = pkg_root / "model" / "ai4i_failure_model.joblib"

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    print("=" * 80)
    print("EVALUATING RANDOM FOREST PREDICTIVE MAINTENANCE MODEL")
    print("=" * 80)
    print(f"Model binary: {model_path}")
    print(f"Operational Decision Threshold: {threshold:.2f}")

    # Load model
    model = joblib.load(model_path)

    # Locate test data if not provided
    if test_features_csv is None or test_labels_csv is None:
        # Search workspace or relative paths
        candidates = [
            pkg_root.parent / "ml" / "data" / "splits",
            pkg_root / "data" / "splits"
        ]
        found_dir = next((c for c in candidates if (c / "X_test.csv").exists()), None)
        if found_dir:
            test_features_csv = found_dir / "X_test.csv"
            test_labels_csv = found_dir / "y_test.csv"
        else:
            print("Note: Test split CSVs not found in local directories. Displaying stored test metrics.")
            summary_path = pkg_root / "evaluation" / "model_performance_summary.json"
            if summary_path.exists():
                with open(summary_path, "r", encoding="utf-8") as f:
                    print(json.dumps(json.load(f), indent=2))
            return

    print(f"Loading test features: {test_features_csv}")
    print(f"Loading test labels:   {test_labels_csv}")
    X_test = pd.read_csv(test_features_csv, index_col=0)[PRIMARY_FEATURES]
    y_test = pd.read_csv(test_labels_csv, index_col=0).squeeze()

    test_proba = model.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, test_pred).ravel()
    acc = accuracy_score(y_test, test_pred)
    prec = precision_score(y_test, test_pred, zero_division=0)
    rec = recall_score(y_test, test_pred, zero_division=0)
    f1 = f1_score(y_test, test_pred, zero_division=0)
    f2 = fbeta_score(y_test, test_pred, beta=2, zero_division=0)
    bal_acc = balanced_accuracy_score(y_test, test_pred)
    roc_auc = roc_auc_score(y_test, test_proba)
    pr_auc = average_precision_score(y_test, test_proba)
    brier = brier_score_loss(y_test, test_proba)

    print("\n--- TEST EVALUATION RESULTS ---")
    print(f"Total Test Records:     {len(y_test):,}")
    print(f"Actual Failures:        {int(y_test.sum())} ({y_test.mean()*100:.2f}%)")
    print(f"Actual Normal Cycles:   {len(y_test) - int(y_test.sum())}")
    print("-" * 50)
    print(f"Accuracy:               {acc*100:.2f}%")
    print(f"Balanced Accuracy:      {bal_acc*100:.2f}%")
    print(f"Failure Recall (Detection Rate): {rec*100:.2f}% ({tp}/{int(y_test.sum())} caught)")
    print(f"Alarm Precision:        {prec*100:.2f}% ({tp}/{tp+fp} alarms were real failures)")
    print(f"F1-Score:               {f1:.4f}")
    print(f"F2-Score (Cost Metric): {f2:.4f}")
    print(f"PR-AUC (Avg Precision): {pr_auc:.4f}")
    print(f"ROC-AUC:                {roc_auc:.4f}")
    print(f"Brier Score Loss:       {brier:.4f}")
    print("-" * 50)
    print("CONFUSION MATRIX:")
    print(f"  True Positives  (Caught Failures): {tp}")
    print(f"  False Positives (False Alarms):    {fp}")
    print(f"  False Negatives (Missed Breakdowns): {fn}")
    print(f"  True Negatives  (Correct Normal):  {tn}")
    print("=" * 80)


if __name__ == "__main__":
    evaluate_dataset()
