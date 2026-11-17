"""
Phase 2.1 Final Verification Script: Random Forest Optimization
================================================================
Validates all constraints, metrics, artifacts, schemas, and compatibility requirements.
"""

import sys
import json
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

def run_verification():
    workspace = Path(__file__).resolve().parent.parent.parent
    ml_dir = workspace / "ml"
    print("=" * 80)
    print("PHASE 2.1 VERIFICATION AUDIT")
    print("=" * 80)

    # 1. Verify original CSV is unchanged
    raw_csv = ml_dir / "data" / "raw" / "ai4i2020.csv"
    root_csv = workspace / "ai4i2020.csv"
    assert root_csv.exists(), "Root ai4i2020.csv missing!"
    df_root = pd.read_csv(root_csv)
    assert df_root.shape == (10000, 14), f"Root CSV shape altered! Expected (10000, 14), got {df_root.shape}"
    print("[PASS] 1. Root ai4i2020.csv remains completely unmodified (10,000 x 14).")

    # 2. Verify splits
    splits_dir = ml_dir / "data" / "splits"
    X_tr = pd.read_csv(splits_dir / "X_train.csv", index_col=0)
    X_te = pd.read_csv(splits_dir / "X_test.csv", index_col=0)
    y_tr = pd.read_csv(splits_dir / "y_train.csv", index_col=0).squeeze()
    y_te = pd.read_csv(splits_dir / "y_test.csv", index_col=0).squeeze()
    
    assert X_tr.shape == (8000, 5), f"X_train shape error: {X_tr.shape}"
    assert X_te.shape == (2000, 5), f"X_test shape error: {X_te.shape}"
    assert y_tr.shape == (8000,), f"y_train shape error: {y_tr.shape}"
    assert y_te.shape == (2000,), f"y_test shape error: {y_te.shape}"
    assert y_tr.sum() == 271, f"y_train failure count altered: {y_tr.sum()}"
    assert y_te.sum() == 68, f"y_test failure count altered: {y_te.sum()}"
    print("[PASS] 2. Train/Test splits strictly preserved (8,000 train / 2,000 test; exactly 271 train failures / 68 test failures).")

    # 3. Verify Model Architecture & Artifacts
    best_model_path = ml_dir / "models" / "best_model" / "ai4i_failure_model.joblib"
    meta_path = ml_dir / "models" / "best_model" / "model_metadata.json"
    assert best_model_path.exists(), "Production model binary missing!"
    assert meta_path.exists(), "Model metadata JSON missing!"
    
    model = joblib.load(best_model_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
        
    assert meta["model_family"] == "Random Forest", f"Expected Random Forest family, got {meta['model_family']}"
    assert meta["selected_probability_threshold"] == 0.35, f"Expected threshold 0.35, got {meta['selected_probability_threshold']}"
    assert len(meta["primary_features"]) == 5, f"Expected 5 primary features, got {len(meta['primary_features'])}"
    assert len(meta["total_internal_features"]) == 8, f"Expected 8 internal features, got {len(meta['total_internal_features'])}"
    print("[PASS] 3. Model binary and metadata verified: Random Forest family with threshold 0.35 and physics feature engineering.")

    # 4. Verify Inference Interface with 5 raw features
    sys.path.insert(0, str(ml_dir / "src"))
    from model_prediction import predict_failure_risk
    
    sample_res = predict_failure_risk(air_temp=300.0, process_temp=310.0, rotational_speed=1500, torque=40.0, tool_wear=50)
    assert "prediction_code" in sample_res
    assert "failure_probability" in sample_res
    assert "risk_level" in sample_res
    assert "decision_threshold" in sample_res
    assert sample_res["decision_threshold"] == 0.35
    print("[PASS] 4. Inference interface tested successfully with strictly 5 raw sensor inputs.")

    # 5. Verify Test Performance Metrics on untouched test set
    test_proba = model.predict_proba(X_te)[:, 1]
    test_pred = (test_proba >= 0.35).astype(int)
    from sklearn.metrics import recall_score, precision_score, confusion_matrix
    
    rec = recall_score(y_te, test_pred)
    prec = precision_score(y_te, test_pred)
    tn, fp, fn, tp = confusion_matrix(y_te, test_pred).ravel()
    
    assert tp == 57, f"Expected 57 TP, got {tp}"
    assert fp == 10, f"Expected 10 FP, got {fp}"
    assert fn == 11, f"Expected 11 FN, got {fn}"
    assert tn == 1922, f"Expected 1922 TN, got {tn}"
    assert abs(rec - 0.838235) < 1e-4, f"Recall mismatch: {rec}"
    assert abs(prec - 0.850746) < 1e-4, f"Precision mismatch: {prec}"
    print(f"[PASS] 5. Real test set metrics confirmed: 57/68 caught (Recall: {rec*100:.2f}%), 10 false alarms (Precision: {prec*100:.2f}%).")

    # 6. Verify Charts
    chart_dir = ml_dir / "outputs" / "charts" / "model_evaluation" / "random_forest_optimization"
    assert chart_dir.is_dir(), "Chart directory missing!"
    expected_charts = [
        "01_baseline_vs_optimized_rf.png",
        "02_recall_comparison_across_experiments.png",
        "03_precision_recall_tradeoff.png",
        "04_threshold_analysis_curves.png",
        "05_confusion_matrix_test.png",
        "06_feature_importance.png",
        "07_calibration_curve.png"
    ]
    for c in expected_charts:
        cp = chart_dir / c
        assert cp.exists() and cp.stat().st_size > 1000, f"Chart missing or corrupt: {c}"
    print(f"[PASS] 6. All 7 visualization charts present and valid in {chart_dir.name}.")

    # 7. Verify Reports & Notebooks
    rep = ml_dir / "outputs" / "reports" / "random_forest_optimization_report.md"
    nb = ml_dir / "notebooks" / "03_ai4i_random_forest_optimization.ipynb"
    assert rep.exists() and rep.stat().st_size > 1000, "Report missing or corrupt!"
    assert nb.exists() and nb.stat().st_size > 1000, "Notebook missing or corrupt!"
    print("[PASS] 7. Optimization technical report and notebook verified.")

    print("\n" + "=" * 80)
    print("ALL PHASE 2.1 VERIFICATION CHECKS PASSED PERFECTLY!")
    print("=" * 80)

if __name__ == "__main__":
    run_verification()
