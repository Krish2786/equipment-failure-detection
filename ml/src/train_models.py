"""
AI4I 2020 Predictive Maintenance - Model Training Pipeline
===========================================================
Trains, cross-validates, tunes decision thresholds, compares, and evaluates
candidate machine learning models on the prepared AI4I 2020 splits.

Models evaluated:
1. Baseline Dummy Classifier (strategy='most_frequent')
2. Decision Tree Classifier (class_weight='balanced')
3. Random Forest Classifier (class_weight='balanced_subsample')
4. XGBoost Classifier (scale_pos_weight adjusted via Pipeline)
"""

import sys
import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple

# Ensure local imports work reliably
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import convert_to_numpy, PRIMARY_FEATURES, TARGET

import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
    brier_score_loss
)

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("ai4i_train")

def resolve_paths() -> Dict[str, Path]:
    current_dir = Path.cwd().resolve()
    if (current_dir / "ml").is_dir():
        ml_dir = current_dir / "ml"
    elif current_dir.name == "ml":
        ml_dir = current_dir
    else:
        ml_dir = Path(__file__).resolve().parent.parent

    paths = {
        "ml_dir": ml_dir,
        "splits_dir": ml_dir / "data" / "splits",
        "candidates_dir": ml_dir / "models" / "candidates",
        "best_model_dir": ml_dir / "models" / "best_model",
        "metrics_dir": ml_dir / "outputs" / "metrics",
        "reports_dir": ml_dir / "outputs" / "reports",
        "charts_dir": ml_dir / "outputs" / "charts" / "model_evaluation"
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths

def load_and_verify_splits(splits_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    logger.info("Loading prepared training and test splits...")
    X_train = pd.read_csv(splits_dir / "X_train.csv", index_col=0)
    X_test = pd.read_csv(splits_dir / "X_test.csv", index_col=0)
    y_train = pd.read_csv(splits_dir / "y_train.csv", index_col=0).squeeze()
    y_test = pd.read_csv(splits_dir / "y_test.csv", index_col=0).squeeze()

    assert X_train.shape == (8000, 5), f"X_train shape error: {X_train.shape}"
    assert X_test.shape == (2000, 5), f"X_test shape error: {X_test.shape}"
    assert len(y_train) == 8000, f"y_train length error: {len(y_train)}"
    assert len(y_test) == 2000, f"y_test length error: {len(y_test)}"

    assert list(X_train.columns) == PRIMARY_FEATURES, f"Columns mismatch: {list(X_train.columns)}"
    assert list(X_test.columns) == PRIMARY_FEATURES, f"Columns mismatch: {list(X_test.columns)}"

    train_counts = y_train.value_counts().to_dict()
    test_counts = y_test.value_counts().to_dict()
    assert train_counts[0] == 7729 and train_counts[1] == 271, f"Train counts unexpected: {train_counts}"
    assert test_counts[0] == 1932 and test_counts[1] == 68, f"Test counts unexpected: {test_counts}"

    logger.info("Splits verified successfully:")
    logger.info(f"  Training: {X_train.shape[0]} samples (Normal: {train_counts[0]}, Failure: {train_counts[1]})")
    logger.info(f"  Testing:  {X_test.shape[0]} samples (Normal: {test_counts[0]}, Failure: {test_counts[1]})")
    return X_train, X_test, y_train, y_test

def create_model_instances(scale_pos_weight: float) -> Dict[str, Any]:
    models = {
        "Dummy Classifier": DummyClassifier(strategy="most_frequent"),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced",
            max_depth=6,
            min_samples_split=8,
            min_samples_leaf=4,
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1
        )
    }
    if XGB_AVAILABLE:
        models["XGBoost"] = Pipeline([
            ("to_numpy", FunctionTransformer(convert_to_numpy)),
            ("classifier", XGBClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1
            ))
        ])
    else:
        logger.warning("XGBoost is not available. Proceeding with remaining algorithms.")
    return models

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray = None) -> Dict[str, float]:
    metrics = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1-Score": float(f1_score(y_true, y_pred, zero_division=0)),
        "F2-Score": float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)),
        "Balanced Accuracy": float(balanced_accuracy_score(y_true, y_pred))
    }
    if y_proba is not None:
        try:
            metrics["ROC-AUC"] = float(roc_auc_score(y_true, y_proba))
        except Exception:
            metrics["ROC-AUC"] = 0.5
        try:
            metrics["PR-AUC"] = float(average_precision_score(y_true, y_proba))
        except Exception:
            metrics["PR-AUC"] = float(y_true.mean())
    else:
        metrics["ROC-AUC"] = 0.5
        metrics["PR-AUC"] = float(y_true.mean())
    return metrics

def run_cross_validation(X_dev: pd.DataFrame, y_dev: pd.Series, models: Dict[str, Any]) -> pd.DataFrame:
    logger.info("Executing 5-Fold Stratified Cross-Validation on development partition (6,400 samples)...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_records = []

    for name in models.keys():
        fold_metrics = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_dev, y_dev), 1):
            X_f_train, X_f_val = X_dev.iloc[train_idx], X_dev.iloc[val_idx]
            y_f_train, y_f_val = y_dev.iloc[train_idx], y_dev.iloc[val_idx]

            if name == "Dummy Classifier":
                m = DummyClassifier(strategy="most_frequent")
            elif name == "Decision Tree":
                m = DecisionTreeClassifier(class_weight="balanced", max_depth=6, min_samples_split=8, min_samples_leaf=4, random_state=42)
            elif name == "Random Forest":
                m = RandomForestClassifier(n_estimators=200, min_samples_split=4, min_samples_leaf=2, class_weight="balanced_subsample", random_state=42, n_jobs=-1)
            elif name == "XGBoost":
                spw = (len(y_f_train) - y_f_train.sum()) / y_f_train.sum()
                m = Pipeline([
                    ("to_numpy", FunctionTransformer(convert_to_numpy)),
                    ("classifier", XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw, eval_metric="logloss", random_state=42, n_jobs=-1))
                ])

            m.fit(X_f_train, y_f_train)
            y_pred = m.predict(X_f_val)
            y_proba = m.predict_proba(X_f_val)[:, 1] if hasattr(m, "predict_proba") else None
            m_res = calculate_metrics(y_f_val.values, y_pred, y_proba)
            m_res["Model"] = name
            m_res["Fold"] = fold
            fold_metrics.append(m_res)

        cv_records.extend(fold_metrics)
        avg_f1 = np.mean([m["F1-Score"] for m in fold_metrics])
        avg_recall = np.mean([m["Recall"] for m in fold_metrics])
        avg_pr_auc = np.mean([m["PR-AUC"] for m in fold_metrics])
        logger.info(f"  [{name}] 5-Fold CV -> PR-AUC: {avg_pr_auc:.4f} | Recall: {avg_recall:.4f} | F1: {avg_f1:.4f}")

    cv_df = pd.DataFrame(cv_records)
    return cv_df

def evaluate_on_validation(
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    models: Dict[str, Any],
    candidates_dir: Path
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    logger.info("Training models on dev partition (6,400) and evaluating on validation partition (1,600)...")
    val_results = []
    fitted_models = {}

    for name, model in models.items():
        model.fit(X_dev, y_dev)
        fitted_models[name] = model

        slug = name.lower().replace(" ", "_")
        joblib.dump(model, candidates_dir / f"{slug}_model.joblib")

        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else None
        res = calculate_metrics(y_val.values, y_pred, y_proba)
        res["Model"] = name
        val_results.append(res)
        logger.info(f"  Validation [{name}] -> PR-AUC: {res['PR-AUC']:.4f} | Recall: {res['Recall']:.4f} | Precision: {res['Precision']:.4f} | F1: {res['F1-Score']:.4f}")

    val_df = pd.DataFrame(val_results)
    return val_df, fitted_models

def run_threshold_tuning(
    model: Any,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    thresholds: list = None
) -> Tuple[pd.DataFrame, float, Dict[str, Any]]:
    if thresholds is None:
        thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]

    y_proba = model.predict_proba(X_val)[:, 1]
    tuning_records = []

    best_threshold = 0.50
    best_f2 = -1.0
    best_stats = {}

    for t in thresholds:
        y_pred_t = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, y_pred_t).ravel()
        p = precision_score(y_val, y_pred_t, zero_division=0)
        r = recall_score(y_val, y_pred_t, zero_division=0)
        f1 = f1_score(y_val, y_pred_t, zero_division=0)
        f2 = fbeta_score(y_val, y_pred_t, beta=2, zero_division=0)

        record = {
            "Threshold": t,
            "Precision": float(p),
            "Recall": float(r),
            "F1-Score": float(f1),
            "F2-Score": float(f2),
            "TP": int(tp),
            "FP": int(fp),
            "FN": int(fn),
            "TN": int(tn)
        }
        tuning_records.append(record)

        if f2 > best_f2:
            best_f2 = f2
            best_threshold = t
            best_stats = record

    tuning_df = pd.DataFrame(tuning_records)
    logger.info(f"Threshold Optimization on Validation Set: Selected Threshold = {best_threshold:.2f} (F2={best_stats['F2-Score']:.4f}, Recall={best_stats['Recall']:.4f}, Precision={best_stats['Precision']:.4f})")
    return tuning_df, best_threshold, best_stats

def retrain_and_evaluate_test(
    best_model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float
) -> Tuple[Any, Dict[str, Any]]:
    logger.info(f"Retraining winning model ({best_model_name}) on full 8,000-sample training dataset...")

    if best_model_name == "Random Forest":
        final_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1
        )
    elif best_model_name == "XGBoost":
        spw = (len(y_train) - y_train.sum()) / y_train.sum()
        final_model = Pipeline([
            ("to_numpy", FunctionTransformer(convert_to_numpy)),
            ("classifier", XGBClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=spw,
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1
            ))
        ])
    elif best_model_name == "Decision Tree":
        final_model = DecisionTreeClassifier(
            class_weight="balanced",
            max_depth=6,
            min_samples_split=8,
            min_samples_leaf=4,
            random_state=42
        )
    else:
        final_model = DummyClassifier(strategy="most_frequent")

    final_model.fit(X_train, y_train)

    logger.info("Evaluating retrained model on held-out test dataset (2,000 samples)...")
    y_proba_test = final_model.predict_proba(X_test)[:, 1] if hasattr(final_model, "predict_proba") else np.zeros(len(y_test))
    
    y_pred_default = final_model.predict(X_test)
    y_pred_opt = (y_proba_test >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_opt).ravel()
    brier = float(brier_score_loss(y_test, y_proba_test))

    test_results = {
        "model_name": best_model_name,
        "selected_threshold": float(threshold),
        "test_dataset_size": len(y_test),
        "test_failure_count": int(y_test.sum()),
        "test_normal_count": int((y_test == 0).sum()),
        "metrics_at_selected_threshold": {
            "Accuracy": float(accuracy_score(y_test, y_pred_opt)),
            "Precision": float(precision_score(y_test, y_pred_opt, zero_division=0)),
            "Recall": float(recall_score(y_test, y_pred_opt, zero_division=0)),
            "F1-Score": float(f1_score(y_test, y_pred_opt, zero_division=0)),
            "F2-Score": float(fbeta_score(y_test, y_pred_opt, beta=2, zero_division=0)),
            "Balanced Accuracy": float(balanced_accuracy_score(y_test, y_pred_opt)),
            "ROC-AUC": float(roc_auc_score(y_test, y_proba_test)),
            "PR-AUC": float(average_precision_score(y_test, y_proba_test)),
            "Brier Score": brier
        },
        "metrics_at_default_0_5": {
            "Accuracy": float(accuracy_score(y_test, y_pred_default)),
            "Precision": float(precision_score(y_test, y_pred_default, zero_division=0)),
            "Recall": float(recall_score(y_test, y_pred_default, zero_division=0)),
            "F1-Score": float(f1_score(y_test, y_pred_default, zero_division=0)),
            "F2-Score": float(fbeta_score(y_test, y_pred_default, beta=2, zero_division=0))
        },
        "confusion_matrix": {
            "True Positives (Detected Failures)": int(tp),
            "False Positives (False Alarms)": int(fp),
            "False Negatives (Missed Failures)": int(fn),
            "True Negatives (Correct Normal)": int(tn)
        },
        "classification_report": classification_report(y_test, y_pred_opt, output_dict=True)
    }

    logger.info("=" * 70)
    logger.info(f"FINAL TEST SET RESULTS [{best_model_name} @ threshold {threshold:.2f}]:")
    logger.info(f"  Recall (Failure Detection Rate): {test_results['metrics_at_selected_threshold']['Recall']*100:.2f}% ({tp}/{y_test.sum()} failures detected)")
    logger.info(f"  Precision:                       {test_results['metrics_at_selected_threshold']['Precision']*100:.2f}%")
    logger.info(f"  F1-Score:                        {test_results['metrics_at_selected_threshold']['F1-Score']:.4f}")
    logger.info(f"  PR-AUC (Average Precision):      {test_results['metrics_at_selected_threshold']['PR-AUC']:.4f}")
    logger.info(f"  ROC-AUC:                         {test_results['metrics_at_selected_threshold']['ROC-AUC']:.4f}")
    logger.info(f"  False Negatives (Missed):        {fn}")
    logger.info(f"  False Positives (False Alarms):  {fp}")
    logger.info("=" * 70)

    return final_model, test_results

def save_best_model_artifacts(
    final_model: Any,
    best_model_name: str,
    threshold: float,
    val_results_row: Dict[str, float],
    test_results: Dict[str, Any],
    paths: Dict[str, Path]
):
    best_dir = paths["best_model_dir"]
    model_path = best_dir / "ai4i_failure_model.joblib"
    joblib.dump(final_model, model_path)
    logger.info(f"Saved best model binary to: {model_path}")

    params = final_model.get_params() if hasattr(final_model, "get_params") else {}
    clean_params = {}
    for k, v in params.items():
        try:
            json.dumps(v)
            clean_params[k] = v
        except (TypeError, OverflowError):
            clean_params[k] = str(v)

    metadata = {
        "model_name": best_model_name,
        "algorithm_class": type(final_model).__name__,
        "model_binary_file": "ai4i_failure_model.joblib",
        "selected_probability_threshold": float(threshold),
        "primary_features": PRIMARY_FEATURES,
        "feature_order": PRIMARY_FEATURES,
        "target_variable": TARGET,
        "target_mapping": {"0": "Normal Operation", "1": "Machine Failure"},
        "random_seed": 42,
        "training_data_info": {
            "total_training_records": 8000,
            "training_normal_count": 7729,
            "training_failure_count": 271,
            "class_imbalance_ratio": "28.52 : 1"
        },
        "validation_metrics": val_results_row,
        "test_metrics": test_results["metrics_at_selected_threshold"],
        "confusion_matrix": test_results["confusion_matrix"],
        "hyperparameters": clean_params,
        "disclaimer": "This model is trained on the AI4I 2020 synthetic predictive maintenance dataset. Output probabilities indicate statistical failure risk under given sensor conditions and do not represent a guaranteed time-to-failure on physical factory machinery."
    }

    metadata_path = best_dir / "model_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    logger.info(f"Saved model metadata to: {metadata_path}")

def run_training_pipeline():
    logger.info("=" * 80)
    logger.info("STARTING MACHINE LEARNING MODEL TRAINING PIPELINE")
    logger.info("=" * 80)

    paths = resolve_paths()
    X_train, X_test, y_train, y_test = load_and_verify_splits(paths["splits_dir"])

    logger.info("Splitting 8,000 training records into 80% dev-train (6,400) and 20% validation (1,600)...")
    X_dev, X_val, y_dev, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.20,
        random_state=42,
        stratify=y_train
    )
    scale_pos_weight = (len(y_dev) - y_dev.sum()) / y_dev.sum()
    logger.info(f"Dev partition: {X_dev.shape[0]} samples (Failures: {y_dev.sum()})")
    logger.info(f"Val partition: {X_val.shape[0]} samples (Failures: {y_val.sum()})")

    models = create_model_instances(scale_pos_weight)

    cv_df = run_cross_validation(X_dev, y_dev, models)
    cv_csv_path = paths["metrics_dir"] / "cross_validation_results.csv"
    cv_df.to_csv(cv_csv_path, index=False)
    logger.info(f"Saved: {cv_csv_path}")

    val_df, fitted_models = evaluate_on_validation(X_dev, y_dev, X_val, y_val, models, paths["candidates_dir"])
    val_csv_path = paths["metrics_dir"] / "validation_results.csv"
    val_df.to_csv(val_csv_path, index=False)
    logger.info(f"Saved: {val_csv_path}")

    cv_summary = cv_df.groupby("Model")[["PR-AUC", "Recall", "Precision", "F1-Score", "ROC-AUC", "Balanced Accuracy"]].mean().reset_index()
    comparison_df = val_df.merge(cv_summary, on="Model", suffixes=("_Validation", "_CV_Mean"))
    comp_csv_path = paths["metrics_dir"] / "model_comparison.csv"
    comparison_df.to_csv(comp_csv_path, index=False)
    logger.info(f"Saved: {comp_csv_path}")

    valid_candidates = val_df[val_df["Model"] != "Dummy Classifier"]
    best_candidate_name = valid_candidates.sort_values(by=["PR-AUC", "F1-Score"], ascending=False).iloc[0]["Model"]
    logger.info(f"--> Algorithm Selected based on Validation Performance: {best_candidate_name}")

    best_fitted_candidate = fitted_models[best_candidate_name]
    tuning_df, optimal_threshold, best_val_stats = run_threshold_tuning(best_fitted_candidate, X_val, y_val)
    tuning_csv_path = paths["metrics_dir"] / "threshold_tuning_validation.csv"
    tuning_df.to_csv(tuning_csv_path, index=False)
    logger.info(f"Saved: {tuning_csv_path}")

    final_model, test_results = retrain_and_evaluate_test(
        best_model_name=best_candidate_name,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        threshold=optimal_threshold
    )

    test_json_path = paths["metrics_dir"] / "test_results.json"
    with open(test_json_path, "w", encoding="utf-8") as f:
        json.dump(test_results, f, indent=4)
    logger.info(f"Saved: {test_json_path}")

    val_row = val_df[val_df["Model"] == best_candidate_name].iloc[0].to_dict()
    save_best_model_artifacts(
        final_model=final_model,
        best_model_name=best_candidate_name,
        threshold=optimal_threshold,
        val_results_row=val_row,
        test_results=test_results,
        paths=paths
    )

    logger.info("=" * 80)
    logger.info("MODEL TRAINING PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)

if __name__ == "__main__":
    run_training_pipeline()
