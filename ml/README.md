# Machine Learning System for Early Detection of Equipment Failure

This repository contains the complete machine learning preprocessing, exploratory data analysis (EDA), model training, evaluation, threshold tuning, Random Forest recall optimization (Phase 2.1), and real-time inference pipeline for the **AI4I 2020 Predictive Maintenance Dataset**.

---

## 1. Directory Structure

```text
ml/
│
├── data/
│   ├── raw/
│   │   └── ai4i2020.csv              # Unmodified copy of the 10,000-record AI4I 2020 dataset
│   ├── processed/
│   │   └── ai4i_cleaned.csv          # Cleaned dataset with explicit data types (10,000 rows x 14 cols)
│   └── splits/
│       ├── X_train.csv               # 8,000 training instances (5 primary features, indexed by UDI)
│       ├── X_test.csv                # 2,000 testing instances (5 primary features, indexed by UDI)
│       ├── y_train.csv               # 8,000 training labels (Machine failure, indexed by UDI)
│       └── y_test.csv                # 2,000 testing labels (Machine failure, indexed by UDI)
│
├── models/
│   ├── best_model/
│   │   ├── ai4i_failure_model.joblib # Serialized production model pipeline (Optimized Random Forest)
│   │   └── model_metadata.json       # Hyperparameters, feature schema, threshold (0.35) & metrics
│   └── candidates/
│       ├── dummy_classifier_model.joblib
│       ├── decision_tree_model.joblib
│       ├── random_forest_model.joblib
│       └── xgboost_model.joblib
│
├── notebooks/
│   ├── 01_ai4i_data_exploration.ipynb          # Step-by-step EDA, data quality & imbalance analysis
│   ├── 02_ai4i_model_training.ipynb            # Phase 2 model training, CV, threshold tuning
│   └── 03_ai4i_random_forest_optimization.ipynb # Phase 2.1 RF optimization, SMOTE, physics engineering
│
├── outputs/
│   ├── charts/
│   │   ├── (01 to 12 EDA charts)               # 12 EDA high-res 300 DPI distribution & relationship plots
│   │   ├── model_evaluation/                   # Phase 2 initial evaluation plots (10 charts)
│   │   └── model_evaluation/random_forest_optimization/ # 7 Phase 2.1 RF optimization charts (300 DPI):
│   │       ├── 01_baseline_vs_optimized_rf.png
│   │       ├── 02_recall_comparison_across_experiments.png
│   │       ├── 03_precision_recall_tradeoff.png
│   │       ├── 04_threshold_analysis_curves.png
│   │       ├── 05_confusion_matrix_test.png
│   │       ├── 06_feature_importance.png
│   │       └── 07_calibration_curve.png
│   │
│   ├── metrics/                                # Performance metrics & comparison tables
│   │   ├── cross_validation_results.csv
│   │   ├── validation_results.csv
│   │   ├── model_comparison.csv
│   │   ├── threshold_tuning_validation.csv
│   │   ├── rf_experiments_comparison.csv       # Phase 2.1 cross-experiment metrics table
│   │   ├── rf_threshold_analysis.csv           # Validation threshold grid (0.05 to 0.75)
│   │   ├── rf_optimized_test_results.json      # Final test set performance metrics
│   │   └── test_results.json                   # Production model test results
│   │
│   └── reports/                                # Detailed Markdown reports & provenance metadata
│       ├── dataset_summary.md                  # Data dictionary, parametric stats & physical units
│       ├── preprocessing_report.md             # Outlier diagnostics, feature selection & split audit
│       ├── feature_analysis.md                 # Physics dynamics & experimental engineered features
│       ├── model_training_report.md            # Phase 2 initial training & evaluation report
│       └── random_forest_optimization_report.md # Phase 2.1 Random Forest optimization technical report
│
├── src/
│   ├── utils.py                                # Shared transformers & physics feature engineering
│   ├── inspect_data.py                         # Dataset audit & telemetry verification script
│   ├── preprocess_data.py                      # Reproducible preprocessing & splitting pipeline
│   ├── eda.py                                  # EDA publication-quality visualization script
│   ├── train_models.py                         # Phase 2 multi-model benchmark training pipeline
│   ├── evaluate_models.py                      # Phase 2 visualization script
│   ├── threshold_analysis.py                   # Phase 2 decision threshold trade-off analysis
│   ├── optimize_random_forest.py               # Phase 2.1 RF optimization suite (Exp A to F)
│   ├── evaluate_rf_optimization.py             # Phase 2.1 visualization generator (7 charts)
│   └── model_prediction.py                    # Production prediction interface & inference test
│
├── requirements.txt                            # Pinned Python package dependencies (incl. imbalanced-learn)
└── README.md                                   # This file
```

---

## 2. Quickstart & Execution Commands

Ensure your Python environment (Python 3.10+) has all dependencies installed:

```bash
pip install -r requirements.txt
```

### A. Run Data Preprocessing Pipeline (Phase 1)
```bash
python ml/src/inspect_data.py
python ml/src/preprocess_data.py
python ml/src/eda.py
```

### B. Run Initial Model Comparison (Phase 2)
```bash
python ml/src/train_models.py
python ml/src/evaluate_models.py
```

### C. Run Phase 2.1 Random Forest Optimization Suite
```bash
# Execute all RF optimization experiments (A: Tuned, C: SMOTE, D: Balanced RF, E: Physics, F: Combined)
python ml/src/optimize_random_forest.py

# Generate all 7 Phase 2.1 publication charts (300 DPI)
python ml/src/evaluate_rf_optimization.py

# Test production model inference on sample telemetry cases
python ml/src/model_prediction.py
```

---

## 3. Production Model: Optimized Random Forest (Phase 2.1)

Per architectural requirements, **Random Forest is the primary production algorithm**. In Phase 2.1, Random Forest was optimized using domain physics feature engineering and operational decision threshold calibration ($\theta = 0.35$).

### 3.1 Final Held-Out Test Set Performance (2,000 Untouched Records)

| Metric | Baseline Random Forest | Secondary Model (XGBoost) | **Optimized Random Forest (Production)** | Improvement vs Baseline |
| :--- | :---: | :---: | :---: | :---: |
| **Failure Detection Recall** | 58.82% (40 / 68 caught) | 82.35% (56 / 68 caught) | **83.82% (57 / 68 caught)** | **+25.00%** |
| **Alarm Precision** | 76.92% (12 false alarms) | 55.45% (45 false alarms) | **85.07% (10 false alarms)** | **+8.15% (+29.6% vs XGBoost)** |
| **F1-Score** | 0.6667 | 0.6627 | **0.8444** | **+0.1777** |
| **F2-Score ($\beta=2$ Safety Metric)** | 0.6173 | 0.7507 | **0.8407** | **+0.2234** |
| **PR-AUC (Average Precision)** | 0.7256 | 0.8001 | **0.8576** | **+0.1320** |
| **ROC-AUC** | 0.9660 | 0.9750 | **0.9791** | **+0.0131** |
| **Overall Accuracy** | 97.80% | 97.15% | **98.95%** | **+1.15%** |
| **Balanced Accuracy** | 79.10% | 90.01% | **91.65%** | **+12.55%** |
| **Brier Score Calibration Loss** | 0.0240 | 0.0221 | **0.0098** | **55.7% lower error** |

### 3.2 Production Test Confusion Matrix (2,000 Records @ Threshold 0.35)
```text
                   Predicted Normal    Predicted Failure
Actual Normal           1,922                 10 (False Alarms)
Actual Failure             11 (Missed)        57 (Caught Failures)
```
- **Caught Failures:** 57 / 68 (83.82% recall).
- **False Alarms:** 10 / 1,932 normal cycles (only 0.52% of normal operations triggered an alarm).

---

## 4. Phase 2.1 Optimization Experiments Summary

Validation results across the 6 explored strategies (on 1,600-sample validation partition):

| Experiment / Configuration | Feature Representation | Class Balancing Strategy | Val Recall | Val Precision | Val F1 | Val F2 | Val PR-AUC | False Alarms |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline RF** | 5 Raw Sensors | `balanced_subsample` | 59.26% | 76.19% | 0.6667 | 0.6202 | 0.7256 | 10 |
| **Exp A: Tuned RF** | 5 Raw Sensors | `balanced_subsample` | 59.26% | 80.00% | 0.6809 | 0.6250 | 0.7289 | 8 |
| **Exp C: SMOTE RF** | 5 Raw Sensors | `SMOTE(strat=0.15)` | 72.22% | 66.10% | 0.6903 | 0.7091 | 0.7489 | 20 |
| **Exp D: Balanced RF** | 5 Raw Sensors | Bootstrap Undersampling | 96.30% | 22.81% | 0.3688 | 0.5856 | 0.6335 | 176 |
| **Exp E: Physics RF** | 8 Features (5 Raw + 3 Physics) | `balanced_subsample` | 70.37% | 86.36% | 0.7755 | 0.7308 | 0.8348 | 6 |
| **Exp F: Combined RF (Selected)** | 8 Features (5 Raw + 3 Physics) | `balanced_subsample` + $\theta=0.35$ | **79.63%** | **84.31%** | **0.8190** | **0.8052** | **0.8348** | **8** |

### Why Physics Feature Engineering Succeeded
Standard Random Forest splits on orthogonal, single-feature thresholds. Machine failures in the AI4I dataset are driven by physical laws with non-linear interactions:
1. **Temperature Difference ($\Delta T = T_{\text{proc}} - T_{\text{air}}$):** Key driver of Heat Dissipation Failure (HDF).
2. **Mechanical Power ($P_{\text{mech}} = \tau \times \omega \times \frac{2\pi}{60}$):** Direct driver of Power Failure (PWF).
3. **Overstrain Index ($OSI = \tau \times \text{wear}$):** Direct driver of Overstrain Failure (OSF).

These 3 derived features account for **48.1% of the total Gini tree split importance**, enabling the model to learn exact failure boundaries rather than approximating curved physical laws with hundreds of step splits.

---

## 5. Deployment Interface & Compatibility

The frontend sends strictly the **5 raw sensor inputs**. The internal pipeline handles physics feature engineering automatically:

```python
from model_prediction import predict_failure_risk

result = predict_failure_risk(
    air_temp=301.5,
    process_temp=310.8,
    rotational_speed=1300,
    torque=65.5,
    tool_wear=215
)
print(result["prediction_label"])    # "Failure Predicted (Action Required)"
print(result["failure_probability"])  # 0.9653
print(result["decision_threshold"])   # 0.35
print(result["risk_level"])           # "CRITICAL"
```

**Risk Severities:**
- **CRITICAL** ($P \ge 0.70$): Immediate spindle shutdown required.
- **HIGH** ($P \ge 0.35$): Schedule preventive maintenance before next batch.
- **MODERATE** ($0.20 \le P < 0.35$): Elevated telemetry; increase monitoring.
- **LOW** ($P < 0.20$): Normal operations; proceed with machining.
