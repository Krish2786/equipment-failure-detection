# Machine Learning System for Early Detection of Equipment Failure
## Production Model Package: Optimized Random Forest Classifier

This repository contains the production-ready machine learning package for the **AI4I 2020 Predictive Maintenance System**. It is designed for team sharing, code walkthroughs, training reproduction, and deployment integration.

---

## 1. Project Overview

Predictive maintenance in modern industrial milling machinery protects multi-million-dollar automated production lines. Missing an impending catastrophic breakdown (a **False Negative**) can result in spindle seizure, shattered cutting tools, and unplanned line downtime exceeding thousands of dollars per hour.

This system monitors real-time telemetry from continuous CNC milling sensors and predicts imminent equipment failure before irreversible mechanical breakdown occurs.

### Production Algorithm Decision
While earlier exploration evaluated several candidate models (Decision Trees, XGBoost), **Random Forest was chosen and optimized as the final production algorithm**. Through domain-physics feature engineering and operational threshold calibration, this Random Forest model achieved **83.82% Failure Detection Recall** with **85.07% Alarm Precision**, outperforming XGBoost and baseline models in both failure capture and false alarm reduction.

---

## 2. Model Architecture & Pipeline Design

The system accepts strictly **5 raw continuous sensor telemetry streams** collected from standard machine sensors:

```text
                  FACTORY SENSORS (5 Raw Inputs)
                  - Air temperature [K]
                  - Process temperature [K]
                  - Rotational speed [rpm]
                  - Torque [Nm]
                  - Tool wear [min]
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│  INTERNAL DOMAIN-PHYSICS FEATURE ENGINEERING                │
│  1. Temperature Difference: ΔT = T_process - T_air          │
│  2. Mechanical Power:       P_mech = Torque * Speed * 2π/60 │
│  3. Overstrain Index:       OSI = Torque * Tool wear        │
└──────────────────────────────┬──────────────────────────────┘
                               │ (8 Telemetry Channels)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│  OPTIMIZED RANDOM FOREST CLASSIFIER                         │
│  - Estimators: 300 Trees                                    │
│  - Max Depth: 15                                            │
│  - Min Samples Split: 4  |  Min Samples Leaf: 2             │
│  - Class Balancing: balanced_subsample                      │
│  - Max Features: sqrt                                       │
└──────────────────────────────┬──────────────────────────────┘
                               │ Continuous Probability P(Failure | x)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│  CALIBRATED OPERATIONAL DECISION THRESHOLD (θ = 0.35)       │
│  - P >= 0.70: CRITICAL (Immediate Spindle Emergency Stop)   │
│  - P >= 0.35: HIGH (Preventive Maintenance Before Next Job) │
│  - P >= 0.20: MODERATE (Elevated Telemetry; Alert Operator) │
│  - P <  0.20: LOW (Nominal Parameters; Continue Milling)    │
└─────────────────────────────────────────────────────────────┘
```

**Zero Frontend Breaking Changes:**  
The external caller and UI frontend continue providing strictly the 5 raw sensor inputs. The feature engineering is completely self-contained within the Scikit-learn Pipeline.

---

## 3. Package Directory Structure

```text
Random_Forest_Equipment_Failure_Model/
│
├── README.md                          # Comprehensive system documentation (this file)
│
├── model/
│   ├── ai4i_failure_model.joblib      # Serialized production Random Forest pipeline
│   └── model_metadata.json            # Hyperparameters, feature schema, threshold & metrics
│
├── src/
│   ├── model_prediction.py            # High-level inference interface & risk categorization
│   ├── feature_engineering.py         # Physics derivation formulas (ΔT, Power, Overstrain)
│   ├── utils.py                       # Compatibility shim for deserialization
│   └── evaluate_model.py              # Test dataset evaluation and confusion matrix script
│
├── training/
│   ├── train_random_forest.py         # Standalone production model training script
│   ├── optimize_random_forest.py      # Complete Phase 2.1 optimization experiment suite
│   ├── threshold_analysis.py          # Operational probability threshold sweep (0.05 - 0.75)
│   └── requirements.txt               # Pinned Python package dependencies
│
├── notebook/
│   └── random_forest_training.ipynb   # 13-step interactive notebook (Google Colab ready)
│
├── evaluation/
│   ├── model_training_report.md       # Phase 2 model comparison report
│   ├── random_forest_optimization_report.md # Phase 2.1 detailed optimization report
│   └── model_performance_summary.json # Verified metrics on the 2,000-sample test set
│
├── examples/
│   ├── sample_input.json              # Sample telemetry payloads (Nominal, Overstrain, Thermal)
│   └── sample_prediction.py           # Demonstration script executing sample predictions
│
└── LICENSE_AND_DATASET_INFO.md        # Dataset provenance (UCI AI4I 2020), CC BY 4.0 license
```

---

## 4. Installation & Environment Setup

Ensure you have **Python 3.10 or higher** installed.

### Step 1: Create a Virtual Environment
```bash
python -m venv venv
```

### Step 2: Activate the Virtual Environment
- **Windows (Command Prompt / PowerShell):**
  ```bash
  venv\Scripts\activate
  ```
- **macOS / Linux:**
  ```bash
  source venv/bin/activate
  ```

### Step 3: Install Required Dependencies
```bash
pip install -r training/requirements.txt
```

---

## 5. Quickstart: Running Predictions

You can run predictions directly using the provided `predict_failure_risk()` interface.

### Python Example:
```python
from src.model_prediction import predict_failure_risk

# Telemetry from an overloaded milling spindle with a worn tool
result = predict_failure_risk(
    air_temp=301.5,           # Air temperature [K]
    process_temp=310.8,       # Process temperature [K]
    rotational_speed=1300.0,   # Rotational speed [rpm]
    torque=65.5,              # Torque [Nm]
    tool_wear=215.0           # Tool wear [min]
)

print("Status:          ", result["prediction_label"])
print("Failure Prob:    ", f"{result['failure_probability']*100:.2f}%")
print("Threshold:       ", result["decision_threshold"])
print("Risk Level:      ", result["risk_level"])
print("Recommended Action:", result["recommended_action"])
```

### Run the Sample Demo:
```bash
python examples/sample_prediction.py
```

Output:
```text
[SCENARIO_01] Nominal Machine Telemetry (Healthy Milling Operation)
  >> PREDICTION RESULT:
     Status:           Normal Operation (Code: 0)
     Failure Prob:     0.00% (Threshold: 0.35)
     Risk Level:       LOW

[SCENARIO_02] Overstrain Failure Profile (High Torque + Worn Tool)
  >> PREDICTION RESULT:
     Status:           Failure Predicted (Action Required) (Code: 1)
     Failure Prob:     96.53% (Threshold: 0.35)
     Risk Level:       CRITICAL
     Action Required:  Immediate spindle shutdown and comprehensive mechanical inspection required.
```

---

## 6. Training Instructions

### Required Dataset
Place `ai4i2020.csv` or prepared data splits (`X_train.csv`, `y_train.csv`) in `data/splits/`. (See `LICENSE_AND_DATASET_INFO.md` for dataset acquisition details).

### A. Retrain Production Random Forest
To retrain the production model and serialize `model/ai4i_failure_model.joblib`:
```bash
python training/train_random_forest.py
```

### B. Run Optimization Experiment Suite
To reproduce the six optimization experiments (Baseline, Tuned, SMOTE, Balanced RF, Physics Features, Combined):
```bash
python training/optimize_random_forest.py
```

### C. Run Threshold Sweep
To evaluate performance across thresholds 0.05 to 0.75:
```bash
python training/threshold_analysis.py
```

### D. Run Google Colab Notebook
Open `notebook/random_forest_training.ipynb` in [Google Colab](https://colab.research.google.com/). The notebook includes automatic dependency installation and file upload dialogs.

---

## 7. Model Evaluation & Verified Performance

The optimized model was evaluated on the **untouched held-out test dataset** (2,000 records containing 68 actual equipment failure events):

### Test Performance Metrics (@ Decision Threshold 0.35):
- **Accuracy:** **98.95%**
- **Failure Detection Recall:** **83.82%** (**57 out of 68 actual failures caught**)
- **Alarm Precision:** **85.07%** (**57 True Positives vs only 10 False Alarms**)
- **F1-Score:** **0.8444**
- **F2-Score (Cost Metric):** **0.8407**
- **PR-AUC (Average Precision):** **0.8576**
- **ROC-AUC:** **0.9791**
- **Balanced Accuracy:** **91.65%**
- **Brier Score Calibration Loss:** **0.0098**

### Test Confusion Matrix (2,000 Total Cycles)
```text
                            PREDICTED
                     Normal (0)    Failure (1)
ACTUAL  Normal (0)      1,922          10        (TN = 1,922 | FP = 10)
        Failure (1)       11           57        (FN = 11    | TP = 57)
```

### Benchmark Comparison:
| Model Configuration | Recall | Precision | F1-Score | F2-Score | False Alarms | Missed Failures |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline Random Forest** | 58.82% | 76.92% | 0.6667 | 0.6173 | 12 | 28 |
| **Secondary Model (XGBoost)** | 82.35% | 55.45% | 0.6627 | 0.7507 | 45 | 12 |
| **Optimized Random Forest (Production)** | **83.82%** | **85.07%** | **0.8444** | **0.8407** | **10** | **11** |

---

## 8. Project Limitations & Industrial Safety Disclaimers

1. **Synthetic Telemetry:** The AI4I 2020 dataset is mathematically simulated from physical equations with overlaid Gaussian noise. While it captures non-linear failure envelopes accurately, real CNC machines exhibit sensor calibration drift, acoustic noise, and tooling chatter not modeled in synthetic data.
2. **Classification vs. Prognostics:** The model classifies impending failure risk under current operating conditions. It does not estimate continuous Remaining Useful Life (RUL) in hours.
3. **Random Failures (RNF):** The 11 missed failures primarily correspond to stochastic random failures (`RNF`), which occur with zero precursor sensor symptoms. Purely random electrical or mechanical events cannot be predicted from sensor telemetry alone.
4. **Safety Verification:** This model is designed as a decision-support and early-warning alarm system. Automated emergency stops should always be coupled with hardware interlocks and physical safety switches.
