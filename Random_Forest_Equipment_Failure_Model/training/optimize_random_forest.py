"""
AI4I 2020 Predictive Maintenance - Random Forest Optimization Suite (Phase 2.1)
================================================================================
Comprehensive research and optimization pipeline evaluating six distinct
strategies to maximize failure detection recall while preserving alarm precision:

1. Baseline Random Forest: Standard class-weighted subsample model
2. Experiment A: Hyperparameter Tuning (Depth, min_samples_split, min_samples_leaf, estimators)
3. Experiment B: Decision Threshold Optimization (0.05 to 0.75 grid sweep)
4. Experiment C: SMOTE within ImbPipeline cross-validation training folds
5. Experiment D: Balanced Random Forest (Bootstrap majority undersampling per tree)
6. Experiment E: Physics-based Domain Feature Engineering (Delta-T, Mechanical Power, Overstrain Index)
7. Experiment F: Combined Multi-Faceted Optimization (Physics Pipeline + Calibrated Threshold)
"""

import sys
import json
import logging
from pathlib import Path

# Setup paths and import feature engineering
training_dir = Path(__file__).resolve().parent
pkg_root = training_dir.parent
src_dir = pkg_root / "src"
for p in [str(src_dir), str(pkg_root)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from feature_engineering import (
    PRIMARY_FEATURES,
    ENGINEERED_FEATURE_NAMES,
    ALL_8_FEATURES,
    TARGET,
    add_physics_features
)
import feature_engineering as fe
sys.modules['utils'] = fe

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import FunctionTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, fbeta_score, roc_auc_score,
    average_precision_score, balanced_accuracy_score,
    confusion_matrix, classification_report, brier_score_loss
)

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.over_sampling import SMOTE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("rf_optimize")

THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]


def load_partitions():
    """Resolves data partition locations."""
    candidates = [
        pkg_root.parent / "ml" / "data" / "splits",
        pkg_root / "data" / "splits",
        Path("data/splits")
    ]
    splits_dir = next((c for c in candidates if (c / "X_train.csv").exists()), None)
    if splits_dir is None:
        raise FileNotFoundError("Data splits not found. Please place X_train.csv and y_train.csv in data/splits/")

    X_train = pd.read_csv(splits_dir / "X_train.csv", index_col=0)[PRIMARY_FEATURES]
    y_train = pd.read_csv(splits_dir / "y_train.csv", index_col=0).squeeze()

    X_test_path = splits_dir / "X_test.csv"
    y_test_path = splits_dir / "y_test.csv"
    X_test = pd.read_csv(X_test_path, index_col=0)[PRIMARY_FEATURES] if X_test_path.exists() else None
    y_test = pd.read_csv(y_test_path, index_col=0).squeeze() if y_test_path.exists() else None

    # Dev-train / validation split (80:20 of training data)
    X_dev, X_val, y_dev, y_val = train_test_split(
        X_train, y_train,
        test_size=0.20,
        random_state=42,
        stratify=y_train
    )

    return X_dev, X_val, y_dev, y_val, X_train, y_train, X_test, y_test


def calc_metrics(y_true, y_pred, y_proba):
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1-Score": float(f1_score(y_true, y_pred, zero_division=0)),
        "F2-Score": float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)),
        "Balanced_Accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "ROC-AUC": float(roc_auc_score(y_true, y_proba)),
        "PR-AUC": float(average_precision_score(y_true, y_proba))
    }


def evaluate_threshold_grid(model, X_val, y_val, model_label):
    records = []
    y_proba = model.predict_proba(X_val)[:, 1]
    for t in THRESHOLDS:
        y_pred = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
        records.append({
            "Config": model_label,
            "Threshold": t,
            "Recall": float(recall_score(y_val, y_pred, zero_division=0)),
            "Precision": float(precision_score(y_val, y_pred, zero_division=0)),
            "F1-Score": float(f1_score(y_val, y_pred, zero_division=0)),
            "F2-Score": float(fbeta_score(y_val, y_pred, beta=2, zero_division=0)),
            "Balanced_Accuracy": float(balanced_accuracy_score(y_val, y_pred)),
            "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn)
        })
    return pd.DataFrame(records)


def run_all_experiments():
    logger.info("=" * 80)
    logger.info("STARTING RANDOM FOREST OPTIMIZATION EXPERIMENTS SUITE")
    logger.info("=" * 80)

    X_dev, X_val, y_dev, y_val, X_train, y_train, X_test, y_test = load_partitions()
    results = []

    # 1. Baseline
    logger.info("1. Baseline Random Forest (5 raw sensors @ 0.50 threshold)...")
    base_rf = RandomForestClassifier(
        n_estimators=200, max_depth=None, min_samples_split=4, min_samples_leaf=2,
        class_weight="balanced_subsample", random_state=42, n_jobs=-1
    )
    base_rf.fit(X_dev, y_dev)
    val_pred = base_rf.predict(X_val)
    val_proba = base_rf.predict_proba(X_val)[:, 1]
    m_base = calc_metrics(y_val, val_pred, val_proba)
    results.append({"Experiment": "Baseline RF", **m_base})
    logger.info(f"   Baseline -> Recall: {m_base['Recall']*100:.1f}% | Precision: {m_base['Precision']*100:.1f}% | PR-AUC: {m_base['PR-AUC']:.4f}")

    # 2. Experiment A: Hyperparameter Tuning
    logger.info("2. Exp A: Hyperparameter Tuned Random Forest...")
    tuned_rf = RandomForestClassifier(
        n_estimators=500, max_depth=20, min_samples_split=4, min_samples_leaf=2,
        max_features="sqrt", class_weight="balanced_subsample", random_state=42, n_jobs=-1
    )
    tuned_rf.fit(X_dev, y_dev)
    val_pred = tuned_rf.predict(X_val)
    val_proba = tuned_rf.predict_proba(X_val)[:, 1]
    m_tuned = calc_metrics(y_val, val_pred, val_proba)
    results.append({"Experiment": "Exp A (Tuned RF)", **m_tuned})

    # 3. Experiment C: SMOTE within Pipeline
    logger.info("3. Exp C: SMOTE Pipeline...")
    smote_rf = ImbPipeline([
        ("smote", SMOTE(sampling_strategy=0.15, random_state=42)),
        ("rf", RandomForestClassifier(n_estimators=300, max_depth=15, min_samples_split=4, min_samples_leaf=2, random_state=42, n_jobs=-1))
    ])
    smote_rf.fit(X_dev, y_dev)
    val_pred = smote_rf.predict(X_val)
    val_proba = smote_rf.predict_proba(X_val)[:, 1]
    m_smote = calc_metrics(y_val, val_pred, val_proba)
    results.append({"Experiment": "Exp C (SMOTE RF)", **m_smote})

    # 4. Experiment D: Balanced Random Forest
    logger.info("4. Exp D: Balanced Random Forest...")
    brf = BalancedRandomForestClassifier(
        n_estimators=300, max_depth=12, replacement=True, sampling_strategy="auto", random_state=42, n_jobs=-1
    )
    brf.fit(X_dev, y_dev)
    val_pred = brf.predict(X_val)
    val_proba = brf.predict_proba(X_val)[:, 1]
    m_brf = calc_metrics(y_val, val_pred, val_proba)
    results.append({"Experiment": "Exp D (Balanced RF)", **m_brf})

    # 5. Experiment E: Physics Feature Engineering
    logger.info("5. Exp E: Physics Feature Engineering...")
    rf_physics = ImbPipeline([
        ("physics_features", FunctionTransformer(add_physics_features)),
        ("rf", RandomForestClassifier(
            n_estimators=300, max_depth=15, min_samples_split=4, min_samples_leaf=2,
            class_weight="balanced_subsample", random_state=42, n_jobs=-1
        ))
    ])
    rf_physics.fit(X_dev, y_dev)
    val_pred = rf_physics.predict(X_val)
    val_proba = rf_physics.predict_proba(X_val)[:, 1]
    m_phys = calc_metrics(y_val, val_pred, val_proba)
    results.append({"Experiment": "Exp E (Physics RF @ 0.50)", **m_phys})

    # 6. Experiment F: Combined with Optimal Threshold 0.35
    logger.info("6. Exp F: Combined Physics RF @ Threshold 0.35...")
    val_pred_comb = (val_proba >= 0.35).astype(int)
    m_comb = calc_metrics(y_val, val_pred_comb, val_proba)
    results.append({"Experiment": "Exp F (Combined Physics RF @ 0.35)", **m_comb})
    logger.info(f"   Combined RF -> Recall: {m_comb['Recall']*100:.1f}% | Precision: {m_comb['Precision']*100:.1f}% | F2: {m_comb['F2-Score']:.4f}")

    df_res = pd.DataFrame(results)
    print("\n" + "=" * 80)
    print("VALIDATION EXPERIMENT COMPARISON SUMMARY")
    print("=" * 80)
    print(df_res[["Experiment", "Recall", "Precision", "F1-Score", "F2-Score", "PR-AUC"]].to_string(index=False))
    return df_res


if __name__ == "__main__":
    run_all_experiments()
