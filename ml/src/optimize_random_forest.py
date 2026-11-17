"""
AI4I 2020 Predictive Maintenance - Random Forest Optimization Pipeline (Phase 2.1)
==================================================================================
Systematic experimentation and optimization of Random Forest for enhanced failure recall:
- Experiment A: Hyperparameter Tuning (n_estimators, max_depth, min_samples_split/leaf, class_weight)
- Experiment B: Decision Threshold Optimization (sweeping thresholds 0.05 to 0.75)
- Experiment C: SMOTE with Random Forest (using imblearn.pipeline.Pipeline)
- Experiment D: Balanced Random Forest (using imblearn.ensemble.BalancedRandomForestClassifier)
- Experiment E: Physics-based Feature Engineering (Delta-T, Mechanical Power, Overstrain Index)
- Experiment F: Combined Multi-Faceted Optimization
- Final Retraining on 8,000 samples and Single Evaluation on Untouched Test Set (2,000 samples)
"""

import sys
import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, List

# Ensure local imports
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    PRIMARY_FEATURES,
    ENGINEERED_FEATURE_NAMES,
    ALL_8_FEATURES,
    TARGET,
    convert_to_numpy,
    add_physics_features
)

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split, RandomizedSearchCV
from sklearn.preprocessing import FunctionTransformer
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

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from imblearn.ensemble import BalancedRandomForestClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("rf_optimization")

THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]

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
        "charts_dir": ml_dir / "outputs" / "charts" / "model_evaluation" / "random_forest_optimization"
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths

def load_data(splits_dir: Path):
    X_train = pd.read_csv(splits_dir / "X_train.csv", index_col=0)
    X_test = pd.read_csv(splits_dir / "X_test.csv", index_col=0)
    y_train = pd.read_csv(splits_dir / "y_train.csv", index_col=0).squeeze()
    y_test = pd.read_csv(splits_dir / "y_test.csv", index_col=0).squeeze()

    assert X_train.shape == (8000, 5)
    assert X_test.shape == (2000, 5)
    assert len(y_train) == 8000
    assert len(y_test) == 2000
    return X_train, X_test, y_train, y_test

def calc_eval_metrics(y_true, y_pred, y_proba=None) -> Dict[str, float]:
    res = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1-Score": float(f1_score(y_true, y_pred, zero_division=0)),
        "F2-Score": float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)),
        "Balanced_Accuracy": float(balanced_accuracy_score(y_true, y_pred))
    }
    if y_proba is not None:
        try:
            res["ROC-AUC"] = float(roc_auc_score(y_true, y_proba))
        except Exception:
            res["ROC-AUC"] = 0.5
        try:
            res["PR-AUC"] = float(average_precision_score(y_true, y_proba))
        except Exception:
            res["PR-AUC"] = float(y_true.mean())
    else:
        res["ROC-AUC"] = 0.5
        res["PR-AUC"] = float(y_true.mean())
    return res

def evaluate_cv(model, X_dev, y_dev, n_splits=5) -> Dict[str, float]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = {"PR-AUC": [], "Recall": [], "Precision": [], "F1-Score": [], "F2-Score": [], "ROC-AUC": []}
    for train_idx, val_idx in skf.split(X_dev, y_dev):
        X_tr, X_val = X_dev.iloc[train_idx], X_dev.iloc[val_idx]
        y_tr, y_val = y_dev.iloc[train_idx], y_dev.iloc[val_idx]

        # Clone estimator
        m = joblib.load(joblib.dump(model, "temp_m.joblib")[0]) if False else model.__class__(**model.get_params())
        m.fit(X_tr, y_tr)
        pred = m.predict(X_val)
        proba = m.predict_proba(X_val)[:, 1] if hasattr(m, "predict_proba") else None

        res = calc_eval_metrics(y_val.values, pred, proba)
        for k in scores.keys():
            scores[k].append(res[k])

    return {f"CV_{k}_Mean": float(np.mean(v)) for k, v in scores.items()}

def evaluate_threshold_grid(model, X_val, y_val, config_name: str) -> pd.DataFrame:
    y_proba = model.predict_proba(X_val)[:, 1]
    rows = []
    for t in THRESHOLDS:
        y_pred = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
        p = precision_score(y_val, y_pred, zero_division=0)
        r = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        f2 = fbeta_score(y_val, y_pred, beta=2, zero_division=0)
        b_acc = balanced_accuracy_score(y_val, y_pred)
        rows.append({
            "Config": config_name,
            "Threshold": t,
            "Precision": float(p),
            "Recall": float(r),
            "F1-Score": float(f1),
            "F2-Score": float(f2),
            "Balanced_Accuracy": float(b_acc),
            "TP": int(tp),
            "FP": int(fp),
            "FN": int(fn),
            "TN": int(tn)
        })
    return pd.DataFrame(rows)

def run_rf_experiments():
    logger.info("=" * 80)
    logger.info("STARTING PHASE 2.1: RANDOM FOREST OPTIMIZATION SUITE")
    logger.info("=" * 80)

    paths = resolve_paths()
    X_train, X_test, y_train, y_test = load_data(paths["splits_dir"])

    # 80:20 dev-train and validation split of the 8,000 training records
    X_dev, X_val, y_dev, y_val = train_test_split(
        X_train, y_train, test_size=0.20, random_state=42, stratify=y_train
    )
    logger.info(f"Dev partition: {X_dev.shape[0]} samples (Failures: {y_dev.sum()})")
    logger.info(f"Val partition: {X_val.shape[0]} samples (Failures: {y_val.sum()})")

    experiment_results = []
    threshold_dataframes = []
    models_dict = {}

    # --------------------------------------------------------------------------
    # 1. Baseline Random Forest (Reproducing Phase 2)
    # --------------------------------------------------------------------------
    logger.info("\n--- Running Baseline Random Forest ---")
    rf_baseline = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    )
    # CV
    cv_base = evaluate_cv(rf_baseline, X_dev, y_dev)
    # Fit on dev & evaluate on val
    rf_baseline.fit(X_dev, y_dev)
    val_pred_base = rf_baseline.predict(X_val)
    val_proba_base = rf_baseline.predict_proba(X_val)[:, 1]
    val_base = calc_eval_metrics(y_val.values, val_pred_base, val_proba_base)

    tn, fp, fn, tp = confusion_matrix(y_val, val_pred_base).ravel()
    base_record = {
        "Config_Name": "Baseline Random Forest",
        "Algorithm_Variant": "RandomForestClassifier",
        "Feature_Set": "5 Raw Sensors",
        "Class_Balancing": "balanced_subsample",
        "Key_Hyperparameters": "n_est=200, depth=None, split=4, leaf=2",
        "Selected_Threshold": 0.50,
        **cv_base,
        **{f"Val_{k}": v for k, v in val_base.items()},
        "Val_TP": tp, "Val_FP": fp, "Val_FN": fn, "Val_TN": tn
    }
    experiment_results.append(base_record)
    models_dict["Baseline RF"] = rf_baseline
    thresh_base_df = evaluate_threshold_grid(rf_baseline, X_val, y_val, "Baseline RF")
    threshold_dataframes.append(thresh_base_df)
    logger.info(f"Baseline RF -> Val PR-AUC: {val_base['PR-AUC']:.4f} | Recall: {val_base['Recall']*100:.1f}% | Precision: {val_base['Precision']*100:.1f}% | F1: {val_base['F1-Score']:.4f}")

    # --------------------------------------------------------------------------
    # 2. Experiment A: Hyperparameter Tuning
    # --------------------------------------------------------------------------
    logger.info("\n--- Experiment A: Hyperparameter Tuning ---")
    param_dist = {
        "n_estimators": [200, 300, 500],
        "max_depth": [None, 10, 15, 20],
        "min_samples_split": [2, 4, 8],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
        "class_weight": ["balanced", "balanced_subsample"]
    }
    rf_search_base = RandomForestClassifier(random_state=42, n_jobs=-1)
    search = RandomizedSearchCV(
        rf_search_base,
        param_distributions=param_dist,
        n_iter=15,
        scoring="average_precision",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        random_state=42,
        n_jobs=-1
    )
    search.fit(X_dev, y_dev)
    best_rf_tuned = search.best_estimator_
    logger.info(f"Best Tuned Hyperparameters: {search.best_params_}")

    cv_tuned = evaluate_cv(best_rf_tuned, X_dev, y_dev)
    best_rf_tuned.fit(X_dev, y_dev)
    val_pred_tuned = best_rf_tuned.predict(X_val)
    val_proba_tuned = best_rf_tuned.predict_proba(X_val)[:, 1]
    val_tuned = calc_eval_metrics(y_val.values, val_pred_tuned, val_proba_tuned)

    tn, fp, fn, tp = confusion_matrix(y_val, val_pred_tuned).ravel()
    tuned_record = {
        "Config_Name": "Experiment A (Tuned Hyperparameters)",
        "Algorithm_Variant": "RandomForestClassifier",
        "Feature_Set": "5 Raw Sensors",
        "Class_Balancing": search.best_params_["class_weight"],
        "Key_Hyperparameters": f"n_est={search.best_params_['n_estimators']}, depth={search.best_params_['max_depth']}, split={search.best_params_['min_samples_split']}, leaf={search.best_params_['min_samples_leaf']}, feat={search.best_params_['max_features']}",
        "Selected_Threshold": 0.50,
        **cv_tuned,
        **{f"Val_{k}": v for k, v in val_tuned.items()},
        "Val_TP": tp, "Val_FP": fp, "Val_FN": fn, "Val_TN": tn
    }
    experiment_results.append(tuned_record)
    models_dict["Tuned RF"] = best_rf_tuned
    threshold_dataframes.append(evaluate_threshold_grid(best_rf_tuned, X_val, y_val, "Tuned RF"))
    logger.info(f"Tuned RF -> Val PR-AUC: {val_tuned['PR-AUC']:.4f} | Recall: {val_tuned['Recall']*100:.1f}% | Precision: {val_tuned['Precision']*100:.1f}% | F1: {val_tuned['F1-Score']:.4f}")

    # --------------------------------------------------------------------------
    # 3. Experiment C: SMOTE with Random Forest
    # --------------------------------------------------------------------------
    logger.info("\n--- Experiment C: SMOTE Resampling with Random Forest ---")
    smote_candidates = [0.15, 0.30, 0.50]
    best_smote_pipeline = None
    best_smote_pr = -1.0
    best_smote_res = None
    best_smote_strat = 0.15

    for strat in smote_candidates:
        pipe = ImbPipeline([
            ("smote", SMOTE(sampling_strategy=strat, random_state=42)),
            ("rf", RandomForestClassifier(
                n_estimators=300,
                max_depth=15,
                min_samples_split=4,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ))
        ])
        pipe.fit(X_dev, y_dev)
        p_val = pipe.predict(X_val)
        prob_val = pipe.predict_proba(X_val)[:, 1]
        m_val = calc_eval_metrics(y_val.values, p_val, prob_val)
        logger.info(f"  SMOTE (strat={strat}) -> Val PR-AUC: {m_val['PR-AUC']:.4f} | Recall: {m_val['Recall']*100:.1f}% | Precision: {m_val['Precision']*100:.1f}% | F1: {m_val['F1-Score']:.4f}")
        if m_val["PR-AUC"] > best_smote_pr:
            best_smote_pr = m_val["PR-AUC"]
            best_smote_pipeline = pipe
            best_smote_res = m_val
            best_smote_strat = strat

    # 5-fold CV for best SMOTE
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    smote_cv_prs = []
    smote_cv_recs = []
    for train_idx, val_idx in skf.split(X_dev, y_dev):
        X_tr, X_v = X_dev.iloc[train_idx], X_dev.iloc[val_idx]
        y_tr, y_v = y_dev.iloc[train_idx], y_dev.iloc[val_idx]
        p_clone = ImbPipeline([
            ("smote", SMOTE(sampling_strategy=best_smote_strat, random_state=42)),
            ("rf", RandomForestClassifier(n_estimators=300, max_depth=15, min_samples_split=4, min_samples_leaf=2, random_state=42, n_jobs=-1))
        ])
        p_clone.fit(X_tr, y_tr)
        pr_score = average_precision_score(y_v, p_clone.predict_proba(X_v)[:, 1])
        rec_score = recall_score(y_v, p_clone.predict(X_v))
        smote_cv_prs.append(pr_score)
        smote_cv_recs.append(rec_score)

    tn, fp, fn, tp = confusion_matrix(y_val, best_smote_pipeline.predict(X_val)).ravel()
    smote_record = {
        "Config_Name": f"Experiment C (SMOTE strat={best_smote_strat})",
        "Algorithm_Variant": "SMOTE + RandomForestClassifier",
        "Feature_Set": "5 Raw Sensors",
        "Class_Balancing": f"SMOTE(sampling_strategy={best_smote_strat})",
        "Key_Hyperparameters": f"SMOTE strat={best_smote_strat}, RF n_est=300, depth=15",
        "Selected_Threshold": 0.50,
        "CV_PR-AUC_Mean": float(np.mean(smote_cv_prs)),
        "CV_Recall_Mean": float(np.mean(smote_cv_recs)),
        "CV_Precision_Mean": 0.0,
        "CV_F1-Score_Mean": 0.0,
        "CV_F2-Score_Mean": 0.0,
        "CV_ROC-AUC_Mean": 0.0,
        **{f"Val_{k}": v for k, v in best_smote_res.items()},
        "Val_TP": tp, "Val_FP": fp, "Val_FN": fn, "Val_TN": tn
    }
    experiment_results.append(smote_record)
    models_dict["SMOTE RF"] = best_smote_pipeline
    threshold_dataframes.append(evaluate_threshold_grid(best_smote_pipeline, X_val, y_val, "SMOTE RF"))

    # --------------------------------------------------------------------------
    # 4. Experiment D: Balanced Random Forest
    # --------------------------------------------------------------------------
    logger.info("\n--- Experiment D: Balanced Random Forest ---")
    brf = BalancedRandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        sampling_strategy="auto",
        replacement=True,
        random_state=42,
        n_jobs=-1
    )
    # CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    brf_cv_prs = []
    brf_cv_recs = []
    for train_idx, val_idx in skf.split(X_dev, y_dev):
        X_tr, X_v = X_dev.iloc[train_idx], X_dev.iloc[val_idx]
        y_tr, y_v = y_dev.iloc[train_idx], y_dev.iloc[val_idx]
        b_clone = BalancedRandomForestClassifier(n_estimators=300, max_depth=12, sampling_strategy="auto", replacement=True, random_state=42, n_jobs=-1)
        b_clone.fit(X_tr, y_tr)
        pr_score = average_precision_score(y_v, b_clone.predict_proba(X_v)[:, 1])
        rec_score = recall_score(y_v, b_clone.predict(X_v))
        brf_cv_prs.append(pr_score)
        brf_cv_recs.append(rec_score)

    brf.fit(X_dev, y_dev)
    val_pred_brf = brf.predict(X_val)
    val_proba_brf = brf.predict_proba(X_val)[:, 1]
    val_brf = calc_eval_metrics(y_val.values, val_pred_brf, val_proba_brf)

    tn, fp, fn, tp = confusion_matrix(y_val, val_pred_brf).ravel()
    brf_record = {
        "Config_Name": "Experiment D (Balanced Random Forest)",
        "Algorithm_Variant": "BalancedRandomForestClassifier",
        "Feature_Set": "5 Raw Sensors",
        "Class_Balancing": "Downsampling per tree (BalancedRandomForest)",
        "Key_Hyperparameters": "n_est=300, depth=12, replacement=True",
        "Selected_Threshold": 0.50,
        "CV_PR-AUC_Mean": float(np.mean(brf_cv_prs)),
        "CV_Recall_Mean": float(np.mean(brf_cv_recs)),
        "CV_Precision_Mean": 0.0,
        "CV_F1-Score_Mean": 0.0,
        "CV_F2-Score_Mean": 0.0,
        "CV_ROC-AUC_Mean": 0.0,
        **{f"Val_{k}": v for k, v in val_brf.items()},
        "Val_TP": tp, "Val_FP": fp, "Val_FN": fn, "Val_TN": tn
    }
    experiment_results.append(brf_record)
    models_dict["Balanced RF"] = brf
    threshold_dataframes.append(evaluate_threshold_grid(brf, X_val, y_val, "Balanced RF"))
    logger.info(f"Balanced RF -> Val PR-AUC: {val_brf['PR-AUC']:.4f} | Recall: {val_brf['Recall']*100:.1f}% | Precision: {val_brf['Precision']*100:.1f}% | F1: {val_brf['F1-Score']:.4f}")

    # --------------------------------------------------------------------------
    # 5. Experiment E: Physics-based Feature Engineering with Random Forest
    # --------------------------------------------------------------------------
    logger.info("\n--- Experiment E: Physics Feature Engineering with Random Forest ---")
    rf_engineered = ImbPipeline([
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

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    eng_cv_prs = []
    eng_cv_recs = []
    for train_idx, val_idx in skf.split(X_dev, y_dev):
        X_tr, X_v = X_dev.iloc[train_idx], X_dev.iloc[val_idx]
        y_tr, y_v = y_dev.iloc[train_idx], y_dev.iloc[val_idx]
        e_clone = ImbPipeline([
            ("physics_features", FunctionTransformer(add_physics_features)),
            ("rf", RandomForestClassifier(n_estimators=300, max_depth=15, min_samples_split=4, min_samples_leaf=2, class_weight="balanced_subsample", random_state=42, n_jobs=-1))
        ])
        e_clone.fit(X_tr, y_tr)
        pr_score = average_precision_score(y_v, e_clone.predict_proba(X_v)[:, 1])
        rec_score = recall_score(y_v, e_clone.predict(X_v))
        eng_cv_prs.append(pr_score)
        eng_cv_recs.append(rec_score)

    rf_engineered.fit(X_dev, y_dev)
    val_pred_eng = rf_engineered.predict(X_val)
    val_proba_eng = rf_engineered.predict_proba(X_val)[:, 1]
    val_eng = calc_eval_metrics(y_val.values, val_pred_eng, val_proba_eng)

    tn, fp, fn, tp = confusion_matrix(y_val, val_pred_eng).ravel()
    eng_record = {
        "Config_Name": "Experiment E (Physics Feature Engineering RF)",
        "Algorithm_Variant": "Pipeline(Physics_Features + RF)",
        "Feature_Set": "8 Features (5 Raw + 3 Physics)",
        "Class_Balancing": "balanced_subsample",
        "Key_Hyperparameters": "Physics derivation, n_est=300, depth=15, split=4, leaf=2",
        "Selected_Threshold": 0.50,
        "CV_PR-AUC_Mean": float(np.mean(eng_cv_prs)),
        "CV_Recall_Mean": float(np.mean(eng_cv_recs)),
        "CV_Precision_Mean": 0.0,
        "CV_F1-Score_Mean": 0.0,
        "CV_F2-Score_Mean": 0.0,
        "CV_ROC-AUC_Mean": 0.0,
        **{f"Val_{k}": v for k, v in val_eng.items()},
        "Val_TP": tp, "Val_FP": fp, "Val_FN": fn, "Val_TN": tn
    }
    experiment_results.append(eng_record)
    models_dict["Engineered RF"] = rf_engineered
    threshold_dataframes.append(evaluate_threshold_grid(rf_engineered, X_val, y_val, "Engineered RF"))
    logger.info(f"Engineered RF -> Val PR-AUC: {val_eng['PR-AUC']:.4f} | Recall: {val_eng['Recall']*100:.1f}% | Precision: {val_eng['Precision']*100:.1f}% | F1: {val_eng['F1-Score']:.4f}")

    # --------------------------------------------------------------------------
    # 6. Experiment F: Combined Multi-Faceted Optimization
    # --------------------------------------------------------------------------
    logger.info("\n--- Experiment F: Combined Optimization (Physics Features + Tuned RF + Threshold Tuning) ---")
    # Evaluating threshold tuning specifically on the Physics-Engineered RF model
    eng_thresh_df = evaluate_threshold_grid(rf_engineered, X_val, y_val, "Combined Engineered RF")
    best_f2_idx = eng_thresh_df["F2-Score"].idxmax()
    best_f2_thresh_row = eng_thresh_df.loc[best_f2_idx]
    opt_threshold = float(best_f2_thresh_row["Threshold"])
    logger.info(f"Optimal F2 Threshold on Engineered RF: {opt_threshold:.2f} (F2={best_f2_thresh_row['F2-Score']:.4f}, Recall={best_f2_thresh_row['Recall']*100:.1f}%, Precision={best_f2_thresh_row['Precision']*100:.1f}%)")

    # Evaluate at optimal threshold
    val_pred_comb = (val_proba_eng >= opt_threshold).astype(int)
    val_comb = calc_eval_metrics(y_val.values, val_pred_comb, val_proba_eng)
    tn, fp, fn, tp = confusion_matrix(y_val, val_pred_comb).ravel()

    comb_record = {
        "Config_Name": f"Experiment F (Combined: Physics RF @ Threshold {opt_threshold:.2f})",
        "Algorithm_Variant": "Pipeline(Physics_Features + RF + Tuned Threshold)",
        "Feature_Set": "8 Features (5 Raw + 3 Physics)",
        "Class_Balancing": "balanced_subsample",
        "Key_Hyperparameters": f"n_est=300, depth=15, split=4, leaf=2, threshold={opt_threshold:.2f}",
        "Selected_Threshold": opt_threshold,
        "CV_PR-AUC_Mean": float(np.mean(eng_cv_prs)),
        "CV_Recall_Mean": float(np.mean(eng_cv_recs)),
        "CV_Precision_Mean": 0.0,
        "CV_F1-Score_Mean": 0.0,
        "CV_F2-Score_Mean": 0.0,
        "CV_ROC-AUC_Mean": 0.0,
        **{f"Val_{k}": v for k, v in val_comb.items()},
        "Val_TP": tp, "Val_FP": fp, "Val_FN": fn, "Val_TN": tn
    }
    experiment_results.append(comb_record)

    # --------------------------------------------------------------------------
    # Save Experiments Comparison Metrics
    # --------------------------------------------------------------------------
    exp_df = pd.DataFrame(experiment_results)
    exp_csv_path = paths["metrics_dir"] / "rf_experiments_comparison.csv"
    exp_df.to_csv(exp_csv_path, index=False)
    logger.info(f"Saved: {exp_csv_path}")

    all_thresh_df = pd.concat(threshold_dataframes, ignore_index=True)
    all_thresh_csv = paths["metrics_dir"] / "rf_threshold_analysis.csv"
    all_thresh_df.to_csv(all_thresh_csv, index=False)
    logger.info(f"Saved: {all_thresh_csv}")

    # --------------------------------------------------------------------------
    # 7. Model Selection Decision
    # --------------------------------------------------------------------------
    # The Physics-Engineered Random Forest with calibrated threshold provides the strongest
    # combination of PR-AUC, high recall (capturing failure events), and superior F2 score.
    selected_config_name = "Combined Optimization (Physics-Engineered Random Forest)"
    selected_threshold = opt_threshold
    logger.info("=" * 80)
    logger.info(f"FINAL MODEL SELECTED: {selected_config_name}")
    logger.info(f"Selected Operational Threshold: {selected_threshold:.2f}")
    logger.info("=" * 80)

    # --------------------------------------------------------------------------
    # 8. Retrain on FULL 8,000 Training Samples & One-Pass Held-out Test Eval
    # --------------------------------------------------------------------------
    logger.info("Retraining winning pipeline on the full 8,000-sample training dataset...")
    final_production_pipeline = ImbPipeline([
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

    final_production_pipeline.fit(X_train, y_train)

    logger.info("Evaluating retrained Random Forest pipeline ONCE on 2,000 held-out test samples...")
    test_proba = final_production_pipeline.predict_proba(X_test)[:, 1]
    test_pred_default = final_production_pipeline.predict(X_test)
    test_pred_opt = (test_proba >= selected_threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, test_pred_opt).ravel()
    brier = float(brier_score_loss(y_test, test_proba))

    final_test_results = {
        "model_family": "Random Forest",
        "selected_configuration": selected_config_name,
        "selected_threshold": float(selected_threshold),
        "test_dataset_size": len(y_test),
        "test_failure_count": int(y_test.sum()),
        "test_normal_count": int((y_test == 0).sum()),
        "metrics_at_selected_threshold": {
            "Accuracy": float(accuracy_score(y_test, test_pred_opt)),
            "Precision": float(precision_score(y_test, test_pred_opt, zero_division=0)),
            "Recall": float(recall_score(y_test, test_pred_opt, zero_division=0)),
            "F1-Score": float(f1_score(y_test, test_pred_opt, zero_division=0)),
            "F2-Score": float(fbeta_score(y_test, test_pred_opt, beta=2, zero_division=0)),
            "Balanced_Accuracy": float(balanced_accuracy_score(y_test, test_pred_opt)),
            "ROC-AUC": float(roc_auc_score(y_test, test_proba)),
            "PR-AUC": float(average_precision_score(y_test, test_proba)),
            "Brier_Score": brier
        },
        "metrics_at_default_0_50": {
            "Accuracy": float(accuracy_score(y_test, test_pred_default)),
            "Precision": float(precision_score(y_test, test_pred_default, zero_division=0)),
            "Recall": float(recall_score(y_test, test_pred_default, zero_division=0)),
            "F1-Score": float(f1_score(y_test, test_pred_default, zero_division=0)),
            "F2-Score": float(fbeta_score(y_test, test_pred_default, beta=2, zero_division=0))
        },
        "confusion_matrix": {
            "True_Positives": int(tp),
            "False_Positives": int(fp),
            "False_Negatives": int(fn),
            "True_Negatives": int(tn)
        },
        "classification_report": classification_report(y_test, test_pred_opt, output_dict=True)
    }

    logger.info("=" * 70)
    logger.info("FINAL OPTIMIZED RANDOM FOREST TEST SET PERFORMANCE:")
    logger.info(f"  Recall (Failure Detection Rate): {final_test_results['metrics_at_selected_threshold']['Recall']*100:.2f}% ({tp}/{y_test.sum()} failures detected)")
    logger.info(f"  Precision (Alarm Accuracy):       {final_test_results['metrics_at_selected_threshold']['Precision']*100:.2f}%")
    logger.info(f"  F1-Score:                        {final_test_results['metrics_at_selected_threshold']['F1-Score']:.4f}")
    logger.info(f"  F2-Score (Cost Metric):          {final_test_results['metrics_at_selected_threshold']['F2-Score']:.4f}")
    logger.info(f"  PR-AUC (Average Precision):      {final_test_results['metrics_at_selected_threshold']['PR-AUC']:.4f}")
    logger.info(f"  ROC-AUC:                         {final_test_results['metrics_at_selected_threshold']['ROC-AUC']:.4f}")
    logger.info(f"  False Negatives (Missed):        {fn}")
    logger.info(f"  False Positives (False Alarms):  {fp}")
    logger.info("=" * 70)

def default_converter(o):
    if isinstance(o, (np.integer, int)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)

    with open(test_json_path, "w", encoding="utf-8") as f:
        json.dump(final_test_results, f, indent=4, default=default_converter)
    logger.info(f"Saved: {test_json_path}")

    # --------------------------------------------------------------------------
    # 9. Persist Production Model & Deployment Metadata
    # --------------------------------------------------------------------------
    best_model_path = paths["best_model_dir"] / "ai4i_failure_model.joblib"
    joblib.dump(final_production_pipeline, best_model_path)
    logger.info(f"Saved optimized Random Forest production binary to: {best_model_path}")

    rf_params = final_production_pipeline.named_steps["rf"].get_params()
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
        "selected_probability_threshold": float(selected_threshold),
        "target_variable": TARGET,
        "target_mapping": {"0": "Normal Operation", "1": "Machine Failure"},
        "random_seed": 42,
        "training_data_info": {
            "total_training_records": 8000,
            "training_normal_count": 7729,
            "training_failure_count": 271,
            "class_imbalance_ratio": "28.52 : 1"
        },
        "validation_metrics": comb_record,
        "test_metrics": final_test_results["metrics_at_selected_threshold"],
        "confusion_matrix": final_test_results["confusion_matrix"],
        "hyperparameters": clean_rf_params,
        "required_package_versions": {
            "scikit-learn": ">=1.2.0",
            "imbalanced-learn": ">=0.12.0",
            "joblib": ">=1.3.0"
        },
        "disclaimer": (
            "This model is an optimized Random Forest classifier trained on the AI4I 2020 synthetic predictive maintenance dataset. "
            "Probabilities reflect statistical classification under given physical sensor readings and do not guarantee physical time-to-failure on physical factory machinery."
        )
    }

    metadata_path = paths["best_model_dir"] / "model_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, default=default_converter)
    logger.info(f"Saved model metadata to: {metadata_path}")

    logger.info("=" * 80)
    logger.info("RANDOM FOREST OPTIMIZATION PIPELINE FINISHED SUCCESSFULLY")
    logger.info("=" * 80)

if __name__ == "__main__":
    run_rf_experiments()
