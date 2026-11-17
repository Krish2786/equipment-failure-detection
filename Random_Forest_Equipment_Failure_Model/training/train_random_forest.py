"""
Production Training Pipeline - Optimized Random Forest Classifier
=================================================================
Trains the final production model for the AI4I 2020 Equipment Failure Detection System.

Architecture:
  - Input: 5 raw continuous machine telemetry sensor streams
  - Step 1: FunctionTransformer(add_physics_features) -> 8 telemetry channels
  - Step 2: RandomForestClassifier(
      n_estimators=300,
      max_depth=15,
      min_samples_split=4,
      min_samples_leaf=2,
      class_weight='balanced_subsample',
      random_state=42,
      n_jobs=-1
    )
  - Calibrated Decision Threshold: 0.35

Artifact Output:
  - model/ai4i_failure_model.joblib
  - model/model_metadata.json
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
from sklearn.preprocessing import FunctionTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, fbeta_score, roc_auc_score,
    average_precision_score, confusion_matrix,
    brier_score_loss, classification_report
)
from imblearn.pipeline import Pipeline as ImbPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("train_rf")


def load_dataset():
    """Locates and loads train/test splits."""
    candidates = [
        pkg_root.parent / "ml" / "data" / "splits",
        pkg_root / "data" / "splits",
        Path("data/splits")
    ]
    splits_dir = next((c for c in candidates if (c / "X_train.csv").exists()), None)
    if splits_dir is None:
        raise FileNotFoundError(
            "Training splits not found. Please ensure X_train.csv and y_train.csv exist in data/splits/"
        )

    logger.info(f"Loading data from: {splits_dir}")
    X_train = pd.read_csv(splits_dir / "X_train.csv", index_col=0)[PRIMARY_FEATURES]
    y_train = pd.read_csv(splits_dir / "y_train.csv", index_col=0).squeeze()

    X_test_path = splits_dir / "X_test.csv"
    y_test_path = splits_dir / "y_test.csv"
    if X_test_path.exists() and y_test_path.exists():
        X_test = pd.read_csv(X_test_path, index_col=0)[PRIMARY_FEATURES]
        y_test = pd.read_csv(y_test_path, index_col=0).squeeze()
    else:
        X_test, y_test = None, None

    return X_train, y_train, X_test, y_test


def train_production_model(save_artifacts: bool = True):
    """
    Trains the winning Random Forest pipeline on the full training set.
    """
    X_train, y_train, X_test, y_test = load_dataset()
    logger.info(f"Training dataset size: {len(X_train)} samples ({y_train.sum()} failures, {len(y_train) - y_train.sum()} normal)")

    # Construct the winning pipeline
    pipeline = ImbPipeline([
        ("physics_features", FunctionTransformer(add_physics_features)),
        ("rf", RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1
        ))
    ])

    logger.info("Training Optimized Random Forest pipeline...")
    pipeline.fit(X_train, y_train)
    logger.info("Model fitting completed successfully!")

    threshold = 0.35

    # If test data is present, evaluate
    test_metrics = {}
    if X_test is not None and y_test is not None:
        logger.info(f"Evaluating on held-out test set ({len(X_test)} samples)...")
        test_proba = pipeline.predict_proba(X_test)[:, 1]
        test_pred = (test_proba >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_test, test_pred).ravel()
        test_metrics = {
            "Accuracy": float(accuracy_score(y_test, test_pred)),
            "Precision": float(precision_score(y_test, test_pred, zero_division=0)),
            "Recall": float(recall_score(y_test, test_pred, zero_division=0)),
            "F1-Score": float(f1_score(y_test, test_pred, zero_division=0)),
            "F2-Score": float(fbeta_score(y_test, test_pred, beta=2, zero_division=0)),
            "ROC-AUC": float(roc_auc_score(y_test, test_proba)),
            "PR-AUC": float(average_precision_score(y_test, test_proba)),
            "Brier_Score": float(brier_score_loss(y_test, test_proba))
        }
        logger.info(f"Test Recall:    {test_metrics['Recall']*100:.2f}% ({tp}/{y_test.sum()} caught)")
        logger.info(f"Test Precision: {test_metrics['Precision']*100:.2f}% ({fp} false alarms)")
        logger.info(f"Test PR-AUC:    {test_metrics['PR-AUC']:.4f}")

    if save_artifacts:
        model_out = pkg_root / "model" / "ai4i_failure_model.joblib"
        meta_out = pkg_root / "model" / "model_metadata.json"
        model_out.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(pipeline, model_out)
        logger.info(f"Saved production model binary to: {model_out}")

        rf_params = pipeline.named_steps["rf"].get_params()
        clean_rf_params = {k: v for k, v in rf_params.items() if isinstance(v, (int, float, str, bool, type(None)))}

        metadata = {
            "model_family": "Random Forest",
            "model_name": "Optimized Random Forest with Physics Feature Engineering",
            "algorithm_class": "Pipeline(FunctionTransformer + RandomForestClassifier)",
            "model_binary_file": "ai4i_failure_model.joblib",
            "primary_features": PRIMARY_FEATURES,
            "feature_order": PRIMARY_FEATURES,
            "internal_derived_features": ENGINEERED_FEATURE_NAMES,
            "total_internal_features": ALL_8_FEATURES,
            "selected_probability_threshold": threshold,
            "target_variable": TARGET,
            "target_mapping": {"0": "Normal Operation", "1": "Machine Failure"},
            "random_seed": 42,
            "training_data_info": {
                "total_training_records": len(X_train),
                "training_normal_count": int(len(X_train) - y_train.sum()),
                "training_failure_count": int(y_train.sum()),
                "class_imbalance_ratio": f"{(len(X_train) - y_train.sum()) / y_train.sum():.2f} : 1"
            },
            "test_metrics": test_metrics,
            "hyperparameters": clean_rf_params,
            "required_package_versions": {
                "scikit-learn": ">=1.2.0",
                "imbalanced-learn": ">=0.12.0",
                "joblib": ">=1.3.0"
            }
        }
        with open(meta_out, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)
        logger.info(f"Saved model metadata to: {meta_out}")

    return pipeline


if __name__ == "__main__":
    train_production_model()
