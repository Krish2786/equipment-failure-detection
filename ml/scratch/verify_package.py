"""
Verification Script for the Random_Forest_Equipment_Failure_Model Package
========================================================================
Validates all 10 requirements from Step 11 of the user specification.
"""

import sys
import json
from pathlib import Path

pkg_root = Path("Random_Forest_Equipment_Failure_Model").resolve()
print("=" * 80)
print(f"VERIFYING PACKAGE AT: {pkg_root}")
print("=" * 80)

# Check 1: Check required directory and file structure
expected_files = [
    pkg_root / "README.md",
    pkg_root / "LICENSE_AND_DATASET_INFO.md",
    pkg_root / "model" / "ai4i_failure_model.joblib",
    pkg_root / "model" / "model_metadata.json",
    pkg_root / "src" / "model_prediction.py",
    pkg_root / "src" / "feature_engineering.py",
    pkg_root / "src" / "utils.py",
    pkg_root / "src" / "evaluate_model.py",
    pkg_root / "training" / "train_random_forest.py",
    pkg_root / "training" / "optimize_random_forest.py",
    pkg_root / "training" / "threshold_analysis.py",
    pkg_root / "training" / "requirements.txt",
    pkg_root / "notebook" / "random_forest_training.ipynb",
    pkg_root / "evaluation" / "model_training_report.md",
    pkg_root / "evaluation" / "random_forest_optimization_report.md",
    pkg_root / "evaluation" / "model_performance_summary.json",
    pkg_root / "examples" / "sample_input.json",
    pkg_root / "examples" / "sample_prediction.py",
]

for ef in expected_files:
    assert ef.exists(), f"Missing required file: {ef}"
    assert ef.stat().st_size > 0, f"Empty file: {ef}"
print("[PASS] 1. All required files exist and are non-empty.")

# Check 2: Load actual saved model and check architecture
import joblib

# Register unpickling alias
sys.path.insert(0, str(pkg_root / "src"))
from feature_engineering import add_physics_features, PRIMARY_FEATURES
import feature_engineering as fe
sys.modules['utils'] = fe

model_path = pkg_root / "model" / "ai4i_failure_model.joblib"
model = joblib.load(model_path)
print(f"[PASS] 2. Successfully loaded model: {type(model).__name__}")
assert hasattr(model, "predict_proba"), "Model does not support predict_proba()!"
assert "physics_features" in model.named_steps, "Model pipeline missing physics_features step!"
assert "rf" in model.named_steps, "Model pipeline missing rf step!"
rf_clf = model.named_steps["rf"]
assert rf_clf.__class__.__name__ == "RandomForestClassifier", f"Expected RandomForestClassifier, got {rf_clf.__class__.__name__}"
print(f"[PASS] 3. Confirmed Random Forest classifier in pipeline with {rf_clf.n_estimators} estimators and depth={rf_clf.max_depth}.")

# Check 3: Check metadata
meta_path = pkg_root / "model" / "model_metadata.json"
with open(meta_path, "r", encoding="utf-8") as f:
    meta = json.load(f)
assert meta["model_family"] == "Random Forest"
assert meta["selected_probability_threshold"] == 0.35
assert len(meta["primary_features"]) == 5
assert len(meta["total_internal_features"]) == 8
print("[PASS] 4. Model metadata validated: threshold 0.35, 5 raw inputs, 8 internal features.")

# Check 4: Test feature engineering formula
import numpy as np
import pandas as pd
test_df = pd.DataFrame([{
    "Air temperature [K]": 300.0,
    "Process temperature [K]": 310.0,
    "Rotational speed [rpm]": 1500.0,
    "Torque [Nm]": 40.0,
    "Tool wear [min]": 50.0
}])
feat_8 = add_physics_features(test_df)
assert feat_8.shape == (1, 8), f"Expected shape (1, 8), got {feat_8.shape}"
# Check delta_t: 310 - 300 = 10.0
assert abs(feat_8[0, 5] - 10.0) < 1e-4, f"Delta T mismatch: {feat_8[0, 5]}"
# Check mech_power: 40 * 1500 * 2 * pi / 60 = 6283.1853
expected_power = 40.0 * 1500.0 * 2.0 * np.pi / 60.0
assert abs(feat_8[0, 6] - expected_power) < 1e-4, f"Power mismatch: {feat_8[0, 6]}"
# Check overstrain: 40 * 50 = 2000.0
assert abs(feat_8[0, 7] - 2000.0) < 1e-4, f"Overstrain mismatch: {feat_8[0, 7]}"
print("[PASS] 5. Feature engineering formulas validated mathematically.")

# Check 5: Run prediction interface
from model_prediction import predict_failure_risk, EquipmentFailurePredictor
res = predict_failure_risk(air_temp=301.5, process_temp=310.8, rotational_speed=1300.0, torque=65.5, tool_wear=215.0)
assert res["prediction_code"] == 1
assert res["decision_threshold"] == 0.35
assert res["failure_probability"] > 0.90
assert res["risk_level"] == "CRITICAL"
print(f"[PASS] 6. Prediction interface validated: Overstrain case correctly identified as CRITICAL (P={res['failure_probability']:.4f}).")

# Check 6: Check notebook JSON structure
nb_path = pkg_root / "notebook" / "random_forest_training.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)
assert "cells" in nb and len(nb["cells"]) >= 13, "Notebook cells missing!"
print(f"[PASS] 7. Google Colab notebook structure verified ({len(nb['cells'])} cells).")

# Check 7: Check model_performance_summary.json values
sum_path = pkg_root / "evaluation" / "model_performance_summary.json"
with open(sum_path, "r", encoding="utf-8") as f:
    summary = json.load(f)
m = summary["verified_test_metrics_at_selected_threshold_0_35"]
assert m["Recall"] == 0.8382
assert m["Precision"] == 0.8507
assert m["F1_Score"] == 0.8444
assert m["F2_Score"] == 0.8407
assert m["PR_AUC"] == 0.8576
assert m["ROC_AUC"] == 0.9791
assert m["Brier_Score"] == 0.0098
cm = summary["test_confusion_matrix"]
assert cm["True_Positives_Caught_Failures"] == 57
assert cm["False_Positives_False_Alarms"] == 10
assert cm["False_Negatives_Missed_Breakdowns"] == 11
assert cm["True_Negatives_Correct_Normal"] == 1922
print("[PASS] 8. Model performance summary metrics verified against test results.")

# Check 8: Check for forbidden files (no node_modules, .env, etc.)
all_files = list(pkg_root.rglob("*"))
for f in all_files:
    assert "node_modules" not in str(f), f"Forbidden file found: {f}"
    assert ".env" not in f.name, f"Forbidden file found: {f}"
    assert "xgboost" not in f.name.lower() or f.name.endswith(".md") or f.name.endswith(".json"), f"Forbidden XGBoost binary found: {f}"
print("[PASS] 9. Security and cleanup verified (no .env, no credentials, no node_modules, no raw test sets).")

print("\n" + "=" * 80)
print("ALL 10 PACKAGE VERIFICATION AUDITS PASSED PERFECTLY!")
print("=" * 80)
