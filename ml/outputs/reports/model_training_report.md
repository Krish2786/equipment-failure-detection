# Machine Learning System for Early Detection of Equipment Failure — Model Training and Evaluation Report (Phase 2)

**Project Title:** Machine Learning System for Early Detection of Equipment Failure  
**Dataset:** AI4I 2020 Predictive Maintenance Dataset  
**Primary Target:** `Machine failure` (Binary: 0 = Normal Operation, 1 = Machine Failure)  
**Execution Timestamp:** 2026-11-01  
**Author:** Antigravity (Senior Machine Learning Engineer & Data Scientist)  

---

## 1. Training Objective
The core objective of Phase 2 is to develop, validate, and compare candidate supervised machine learning algorithms to detect imminent milling equipment failures from physical sensor telemetry.

In industrial predictive maintenance, the design criterion is fundamentally asymmetric:
- **Cost of a False Negative (Missed Failure):** Catastrophic mechanical breakdown, unplanned factory downtime, broken tooling, secondary workpiece damage, and expensive emergency repairs.
- **Cost of a False Positive (False Alarm):** A scheduled inspection by a maintenance technician, localized bearing check, or tool inspection—an order of magnitude cheaper than catastrophic line seizure.

Therefore, models in this project are optimized for **Failure Recall**, **Precision-Recall AUC (PR-AUC)**, and **$F_2$-Score**, rather than naive accuracy.

---

## 2. Dataset Summary & Partitioning Provenance

The data splits established in Phase 1 were strictly preserved:
- **Raw Dataset:** 10,000 synthetic records modeled on real-world industrial CNC milling telemetry.
- **Primary Training Partition (`X_train`, `y_train`):** **8,000 instances**
  - Normal Operations (0): 7,729 (96.61%)
  - Machine Breakdowns (1): 271 (3.39%)
- **Held-Out Test Partition (`X_test`, `y_test`):** **2,000 instances**
  - Normal Operations (0): 1,932 (96.60%)
  - Machine Breakdowns (1): 68 (3.40%)
- **Leakage Prevention:** `UDI` is maintained purely as an index for row-level traceability and was never fed into any model.
- **Model Development Strategy:** The 8,000 training records were partitioned into an 80% development set (**6,400 records**, 217 failures) and a 20% validation set (**1,600 records**, 54 failures). The 2,000-record test set remained strictly sealed and was evaluated **exactly once** after all model selections and threshold tuning were finalized.

---

## 3. Feature and Target Schema

The models receive exactly the 5 continuous physical sensor features matching the web application form:

| Feature Name | Type | Physical Unit | Description & Engineering Significance |
| :--- | :--- | :--- | :--- |
| `Air temperature [K]` | Float64 | Kelvin ($K$) | Ambient room temperature around the milling machine ($~300.0$ K). |
| `Process temperature [K]` | Float64 | Kelvin ($K$) | Temperature at the tool-workpiece interface ($~310.0$ K, $\sim 10$ K thermal offset). |
| `Rotational speed [rpm]` | Int64 | Revolutions/min | Spindle angular velocity derived from a 2,860 W nominal motor load. |
| `Torque [Nm]` | Float64 | Newton-meters ($Nm$) | Resistance torque exerted on the cutting spindle (mean $\sim 40$ Nm). |
| `Tool wear [min]` | Int64 | Minutes | Accumulated milling duration of the cutting insert before replacement. |

**Excluded Attributes:**
- `UDI`: Arbitrary identifier (preventing ordering bias).
- `Product ID` & `Type`: Identifier and variant code (ensuring 1:1 frontend compatibility).
- `TWF`, `HDF`, `PWF`, `OSF`, `RNF`: Diagnostic failure sub-modes. **Excluded to prevent direct target leakage.**

---

## 4. Class Imbalance Mitigation Strategy

The dataset exhibits a severe **28.5 : 1 class imbalance** (96.61% normal vs 3.39% failure). Rather than applying synthetic oversampling (SMOTE) or deleting legitimate majority records, class imbalance was addressed via algorithmic loss reweighting:
1. **Decision Tree:** `class_weight='balanced'` — adjusts loss inversely proportional to class frequencies:
   $$w_1 = \frac{N}{2 \times N_1} \approx \frac{6400}{2 \times 217} \approx 14.75$$
2. **Random Forest:** `class_weight='balanced_subsample'` — dynamically recomputes class weighting for each bootstrap sample to handle local density variations.
3. **XGBoost:** `scale_pos_weight` — controls the gradient step weight for positive failure instances:
   $$\text{scale\_pos\_weight} = \frac{N_{\text{negative}}}{N_{\text{positive}}} = \frac{6,183}{217} \approx 28.49 \quad (\text{dev-train}), \quad \frac{7,729}{271} \approx 28.52 \quad (\text{full-train})$$

---

## 5. Candidate Algorithms & Hyperparameters

| Model ID | Algorithm | Purpose / Role | Core Hyperparameters |
| :--- | :--- | :--- | :--- |
| **Model 1** | `DummyClassifier` | Majority baseline | `strategy='most_frequent'` |
| **Model 2** | `DecisionTreeClassifier` | Interpretable tree baseline | `class_weight='balanced'`, `max_depth=6`, `min_samples_split=8`, `min_samples_leaf=4`, `random_state=42` |
| **Model 3** | `RandomForestClassifier` | Ensemble candidate | `n_estimators=200`, `max_depth=None`, `min_samples_split=4`, `min_samples_leaf=2`, `class_weight='balanced_subsample'`, `n_jobs=-1`, `random_state=42` |
| **Model 4** | `XGBClassifier` | Gradient boosting candidate | `n_estimators=200`, `max_depth=5`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`, `scale_pos_weight=28.52`, `eval_metric='logloss'`, `n_jobs=-1`, `random_state=42` |

---

## 6. Model Validation & Cross-Validation Results

### 6.1 5-Fold Stratified Cross-Validation on Development Set (6,400 samples)
Stratified 5-Fold CV was executed on the dev-train partition. Folds maintained identical class distributions.

| Model Candidate | Mean Accuracy | Mean PR-AUC | Mean Recall | Mean Precision | Mean F1-Score | Mean ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Dummy Classifier** | 96.61% | 0.0339 | 0.00% | 0.00% | 0.0000 | 0.5000 |
| **Decision Tree** | 93.28% | 0.5905 | 81.54% | 31.62% | 0.4963 | 0.9083 |
| **Random Forest** | **98.02%** | **0.7273** | 56.64% | **74.81%** | **0.6418** | 0.9668 |
| **XGBoost** | 96.52% | 0.7186 | **76.45%** | 54.12% | 0.6319 | **0.9705** |

### 6.2 Evaluation on Validation Partition (1,600 samples)

| Model Candidate | Validation Accuracy | Validation PR-AUC | Validation Recall | Validation Precision | Validation F1 | Validation F2 | Validation ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Dummy Classifier** | 96.62% | 0.0338 | 0.00% | 0.00% | 0.0000 | 0.0000 | 0.5000 |
| **Decision Tree** | 92.38% | 0.4455 | 74.07% (40/54) | 30.77% | 0.4348 | 0.5831 | 0.8877 |
| **Random Forest** | **98.06%** | 0.7256 | 59.26% (32/54) | **76.19%** | **0.6667** | 0.6190 | 0.9654 |
| **XGBoost** | 96.44% | **0.7324** | **75.93% (41/54)** | 56.94% | 0.6508 | **0.7118** | **0.9702** |

---

## 7. Decision Threshold Optimization Analysis

Using the validation partition, decision thresholds from 0.10 to 0.75 were systematically evaluated:

| Threshold ($t$) | Precision | Recall (Failures Caught) | F1-Score | F2-Score (Cost Metric) | False Alarms (FP) | Missed Breakdowns (FN) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.10** | 22.8% | 94.4% (51/54) | 0.3670 | 0.5876 | 172 | 3 |
| **0.20** | 35.6% | 87.0% (47/54) | 0.5054 | 0.6772 | 85 | 7 |
| **0.30** | 44.8% | 81.5% (44/54) | 0.5789 | 0.7006 | 54 | 10 |
| **0.40** | 50.6% | 75.9% (41/54) | 0.6074 | 0.6902 | 40 | 13 |
| **0.50 (Selected)**| **56.9%** | **75.9% (41/54)** | **0.6508** | **0.7118** | **31** | **13** |
| **0.60** | 62.1% | 66.7% (36/54) | 0.6429 | 0.6569 | 22 | 18 |
| **0.70** | 73.0% | 50.0% (27/54) | 0.5934 | 0.5336 | 10 | 27 |

### Operational Recommendation:
- **Selected Threshold:** **0.50** achieves the peak **$F_2$-Score of 0.7118**, capturing **75.93% of breakdowns** while generating only 31 false alarms across 1,600 operating hours.
- **High-Sensitivity Setting (Optional):** If the factory environment imposes zero-tolerance on missed failures, lowering the threshold to **0.25** increases failure capture to **83.3%** at the expense of ~65 false alarms.

---

## 8. Final Model Selection Rationale

**Selected Model:** **XGBoost Classifier (`Pipeline`)**  
**Selected Threshold:** **0.50**

### Comparative Decision Analysis:
1. **Superior Failure Recall:** At threshold 0.50, XGBoost detects **75.93% of validation failures**, whereas Random Forest only detects **59.26%** (missing 22 out of 54 failures).
2. **Top PR-AUC:** XGBoost achieved the highest Precision-Recall AUC (**0.7324** validation, **0.8001** on held-out test), outperforming Random Forest (0.7256) and Decision Tree (0.4455).
3. **Cost-Optimal $F_2$-Score:** XGBoost attained an $F_2$-Score of **0.7118** compared to Random Forest's **0.6190**, directly reflecting superior industrial cost performance.
4. **Generalization Stability:** The delta between 5-fold CV PR-AUC (0.7186) and validation PR-AUC (0.7324) is minimal ($+0.0138$), indicating robust generalization without overfitting.

---

## 9. Final Evaluation on Untouched Test Dataset (2,000 samples)

The selected XGBoost pipeline was retrained on the **full 8,000 training records** and evaluated **exactly once** on the held-out 2,000-record test set:

| Evaluation Metric | Test Score | Engineering Significance |
| :--- | :--- | :--- |
| **Overall Accuracy** | **97.15%** | Correct classification rate across both normal and failure states. |
| **Failure Recall ($R$)** | **82.35%** | **56 out of 68 actual equipment failures successfully detected.** |
| **Precision ($P$)** | **55.45%** | 56 out of 101 generated failure alarms are true impending failures. |
| **F1-Score** | **0.6627** | Harmonic mean balancing alarm precision and failure detection. |
| **F2-Score** | **0.7507** | Recall-emphasized industrial safety and cost performance score. |
| **Balanced Accuracy** | **90.01%** | Unweighted average of normal specificity (97.67%) and failure recall (82.35%). |
| **ROC-AUC** | **0.9750** | Strong discriminative ability across all operating thresholds. |
| **PR-AUC (Average Precision)**| **0.8001** | **23.6x higher than the no-skill random baseline (0.0340).** |
| **Brier Score Loss** | **0.0222** | Mean squared difference between predicted probability and actual class. |

### Test Confusion Matrix

| | Predicted: Normal (`0`) | Predicted: Failure (`1`) | Total Actual |
| :--- | :---: | :---: | :---: |
| **Actual: Normal (`0`)** | **1,887 (True Negatives)** | **45 (False Positives / False Alarms)** | 1,932 |
| **Actual: Failure (`1`)** | **12 (False Negatives / Missed)** | **56 (True Positives / Caught)** | 68 |
| **Total Predicted** | 1,899 | 101 | **2,000** |

---

## 10. Feature Importance and Interpretability

Native Gain-based feature importance for the retrained model:

| Sensor Feature | Relative Importance Weight | Dominant Associated Failure Mode |
| :--- | :---: | :--- |
| **`Torque [Nm]`** | **38.8%** | Overstrain Failure (`OSF`), Power Failure (`PWF`) |
| **`Rotational speed [rpm]`** | **27.4%** | Heat Dissipation Failure (`HDF`), Power Failure (`PWF`) |
| **`Tool wear [min]`** | **18.2%** | Tool Wear Failure (`TWF`), Overstrain Failure (`OSF`) |
| **`Process temperature [K]`**| **8.9%** | Heat Dissipation Failure (`HDF`) |
| **`Air temperature [K]`** | **6.7%** | Ambient cooling gradient dissipation |

### Physical Interpretation:
- **Torque and Spindle Speed dominate $(>66\%)$ predictive power:** Mechanical work $P = \tau \cdot \omega$ and mechanical stress govern the most violent failure modes (stalls, runaway spindle overspeed, and bearing seizure).
- **Tool wear accounts for 18.2%:** Degradation of micro-cutting geometry steadily increases resistance forces, directly triggering tool breakage beyond 200 minutes.
- **Temperature metrics provide secondary thermal thresholds:** Ambient and cutting zone temperatures serve as thermal tripwires for heat dissipation limits.
- **Important Note:** Feature importance indicates statistical predictive contribution within the model, not direct thermodynamic proof of physical causation.

---

## 11. Probability Calibration Analysis
- **Reliability Diagram:** Evaluated via Scikit-learn's `calibration_curve` across 8 probability bins.
- **Brier Score:** **0.0222**, demonstrating that predicted failure probabilities closely align with empirical failure frequencies.
- **Behavior:** In the critical range $P \in [0.40, 0.85]$, the model shows slight conservative under-confidence, meaning when the model reports a 60% probability, the true empirical breakdown risk is closer to 65–70%. This behavior is favorable for preventive maintenance scheduling.

---

## 12. Artifacts and Production Assets Created

```text
ml/
├── models/
│   ├── best_model/
│   │   ├── ai4i_failure_model.joblib   # Serialized production pipeline
│   │   └── model_metadata.json         # Deployment metadata & schema specification
│   └── candidates/
│       ├── dummy_classifier_model.joblib
│       ├── decision_tree_model.joblib
│       ├── random_forest_model.joblib
│       └── xgboost_model.joblib
│
├── outputs/
│   ├── charts/model_evaluation/        # 10 High-res 300 DPI evaluation figures
│   │   ├── 01_model_comparison_validation.png
│   │   ├── 02_cross_validation_performance.png
│   │   ├── 03_confusion_matrix_test.png
│   │   ├── 04_roc_curve_comparison.png
│   │   ├── 05_precision_recall_curve.png
│   │   ├── 06_feature_importance.png
│   │   ├── 07_threshold_vs_precision_recall.png
│   │   ├── 08_threshold_vs_f1_f2.png
│   │   ├── 09_calibration_curve.png
│   │   └── 10_train_vs_validation_performance.png
│   │
│   └── metrics/
│       ├── cross_validation_results.csv
│       ├── validation_results.csv
│       ├── model_comparison.csv
│       ├── threshold_tuning_validation.csv
│       └── test_results.json
```

---

## 13. System Limitations & Industrial Engineering Context
1. **Synthetic Nature of AI4I 2020:** The dataset is generated through physical milling modeling combined with random-walk noise. While it captures physical thresholds ($\Delta T < 8.6$ K, power bounds, etc.), it does not incorporate real-world factory vibration transients, sensor drift, acoustic emissions, or intermittent electrical dropouts.
2. **Classification vs Remaining Useful Life (RUL):** This model classifies whether current telemetry conditions correspond to imminent or active failure conditions. It does not output a continuous time-to-failure (e.g., "7.4 operating hours remaining").
3. **Exclusion of Equipment Type:** The baseline model was trained strictly on the 5 physical sensors to guarantee seamless compatibility with the existing user interface.

---

## 14. Conclusion & Next Steps
Phase 2 is complete. The system has:
- A thoroughly validated, reproducible training and cross-validation pipeline.
- An **82.35% failure detection rate** with an **average precision (PR-AUC) of 0.8001** on completely held-out data.
- A functional Python prediction script (`src/model_prediction.py`) ready for API integration.

**Recommended Next Step:** Phase 3 — Backend API Development (FastAPI or Flask service) to bridge the serialized model with the existing React frontend interface.
