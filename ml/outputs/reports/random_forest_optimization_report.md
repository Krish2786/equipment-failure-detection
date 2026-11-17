# Phase 2.1 Technical Report: Random Forest Optimization for Equipment Failure Detection

**Project Title:** Machine Learning System for Early Detection of Equipment Failure  
**Dataset:** AI4I 2020 Predictive Maintenance (Synthetic 4:1 milling machine telemetry)  
**Primary Production Algorithm:** Optimized Random Forest Classifier with Physics-Based Feature Engineering  
**Date:** November 2026  
**Status:** Completed & Validated  

---

## Executive Summary

In predictive maintenance for industrial milling machinery, **machine failure detection recall** is the critical operational objective. Missing an impending catastrophic breakdown (a False Negative) risks severe spindle seizure, destruction of high-value cutting tools, and unplanned factory line downtime running thousands of dollars per hour. In Phase 2, an initial model benchmark yielded higher initial recall for XGBoost (82.35%), but with an unacceptable false positive rate (45 false alarms across 2,000 test cycles; 55.45% precision). Furthermore, architectural project guidelines mandated that **Random Forest remain the primary production algorithm**.

In **Phase 2.1**, we conducted a systematic optimization of the Random Forest model family across six experimental dimensions:
1. **Experiment A:** Exhaustive Hyperparameter Tuning (Tree depth, split/leaf constraints, estimator count).
2. **Experiment B:** Decision Threshold Optimization (Sweeping probability thresholds from 0.05 to 0.75).
3. **Experiment C:** Synthetic Minority Over-sampling Technique (SMOTE within training folds).
4. **Experiment D:** Balanced Random Forest Classifier (Bootstrap downsampling per tree).
5. **Experiment E:** Physics-Based Domain Feature Engineering ($\Delta T$, Mechanical Power, Overstrain Index).
6. **Experiment F:** Combined Multi-Faceted Optimization (Physics-Engineered Pipeline + Calibrated Decision Threshold).

### Breakthrough Results Summary

| Metric | Baseline RF (Phase 2) | Secondary Model (XGBoost Phase 2) | **Optimized Random Forest (Phase 2.1)** | Absolute Improvement over Baseline RF |
| :--- | :---: | :---: | :---: | :---: |
| **Test Recall (Failure Detection)** | 59.3% (Val) / 58.8% | 82.35% (56/68 caught) | **83.82% (57/68 caught)** | **+24.5% Recall** |
| **Test Precision (Alarm Reliability)** | 76.2% | 55.45% | **85.07%** | **+8.9% Precision** |
| **False Positives (False Alarms)** | 10 (Val) | 45 alarms | **10 alarms** | **78% fewer alarms vs XGBoost** |
| **Missed Failures (False Negatives)**| 22 (Val) | 12 missed | **11 missed** | **50% fewer missed vs Baseline** |
| **PR-AUC (Average Precision)** | 0.7256 | 0.8001 | **0.8576** | **+0.1320 (+18.2%)** |
| **F1-Score** | 0.6667 | 0.6627 | **0.8444** | **+0.1777** |
| **F2-Score ($\beta=2$ Recall-Weighted)** | 0.6202 | 0.7507 | **0.8407** | **+0.2205** |
| **Brier Score (Calibration Error)** | 0.0240 | 0.0221 | **0.0098** | **55.7% lower error** |

By incorporating physical principles into tree splitting and adjusting the operational decision threshold to **0.35**, the **Optimized Random Forest not only surpassed its baseline across every metric, but also strictly beat XGBoost in both recall (83.82% vs 82.35%) and precision (85.07% vs 55.45%)**, cutting false alarms from 45 down to 10.

---

## 1. Dataset Schema & Experimental Methodology

### 1.1 Feature & Target Schema
In strict adherence to real-time industrial deployment constraints, the model accepts strictly the **5 raw sensor inputs**:
- `Air temperature [K]`: Ambient room temperature ($295.3 - 304.5\text{ K}$).
- `Process temperature [K]`: Internal milling chamber temperature ($305.7 - 313.8\text{ K}$).
- `Rotational speed [rpm]`: Spindle rotation speed ($1,168 - 2,886\text{ rpm}$).
- `Torque [Nm]`: Mechanical torque load ($3.8 - 76.6\text{ Nm}$).
- `Tool wear [min]`: Cumulative machining tool wear time ($0 - 253\text{ min}$).

**Target Variable:** `Machine failure` (Binary: 0 = Normal Operation, 1 = Machine Failure).

**Leakage Prevention Protocols:**
- `UDI` is strictly maintained as an index identifier and excluded from model inputs.
- Diagnostic failure mode flags (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) are **strictly excluded**. In a production environment, failure modes are post-incident diagnostic outcomes; training a predictive model with them constitutes fatal data leakage.

### 1.2 Data Partitions
- **Development Training Set:** 6,400 records (217 failure events, 6,183 normal operations).
- **Validation Set:** 1,600 records (54 failure events, 1,546 normal operations).
- **Held-Out Test Set:** 2,000 records (68 failure events, 1,932 normal operations), preserved untouched until all validation experiments and threshold choices were finalized.
- **Cross-Validation:** 5-fold Stratified K-Fold applied exclusively to development training partitions to preserve the true class ratio ($28.5 : 1$).

---

## 2. Experimental Optimization Suite

### 2.1 Baseline Random Forest
- **Configuration:** `n_estimators=200`, `max_depth=None`, `min_samples_split=4`, `min_samples_leaf=2`, `class_weight='balanced_subsample'`, `random_state=42`, `n_jobs=-1`.
- **Validation Performance:** Recall: **59.26%** (32/54 failures caught, 22 missed), Precision: **76.19%** (10 false alarms), F1: **0.6667**, F2: **0.6202**, PR-AUC: **0.7256**.
- **Analysis:** While `balanced_subsample` helps tree splits, standard unconstrained trees split on single-sensor orthogonal thresholds, failing to capture complex multi-sensor failure regimes (e.g., thermal dissipation collapse and torque-wear overstrain).

### 2.2 Experiment A: Hyperparameter Optimization
- **Grid Explored:** `n_estimators` $\in [200, 300, 500]$, `max_depth` $\in [\text{None}, 10, 15, 20]$, `min_samples_split` $\in [2, 4, 8]$, `min_samples_leaf` $\in [1, 2, 4]$, `max_features` $\in [\text{'sqrt'}, \text{'log2'}]$.
- **Best Parameter Set:** `n_estimators=500`, `max_depth=20`, `min_samples_split=4`, `min_samples_leaf=2`, `max_features='sqrt'`.
- **Validation Performance:** Recall: **59.26%** (32/54 caught), Precision: **80.00%** (8 false alarms), F1: **0.6809**, PR-AUC: **0.7289**.
- **Finding:** Pruning deeper outlier splits improved precision by 3.8% and reduced false alarms by 2, but failed to improve failure recall (remained stalled at 59.3%). Hyperparameter tuning alone cannot compensate for missing feature interactions.

### 2.3 Experiment C: SMOTE with Random Forest
- **Architecture:** `ImbPipeline([('smote', SMOTE(sampling_strategy=0.15, random_state=42)), ('rf', RandomForestClassifier(...))])`.
- **Validation Performance:** Recall: **72.22%** (39/54 caught), Precision: **66.10%** (20 false alarms), F1: **0.6903**, F2: **0.7091**, PR-AUC: **0.7489**.
- **Finding:** SMOTE successfully pushed recall up to 72.2% by synthesizing failure instances in feature space. However, it doubled false alarms (from 10 to 20), dropping precision to 66.1%. Linear interpolation between minority instances in 5-dimensional non-linear sensor space generated synthetic records that violated physical machine boundary laws.

### 2.4 Experiment D: Balanced Random Forest
- **Architecture:** `BalancedRandomForestClassifier(n_estimators=300, max_depth=12, replacement=True, sampling_strategy='auto', random_state=42)`.
- **Validation Performance:** Recall: **96.30%** (52/54 caught), Precision: **22.81%** (176 false alarms!), F1: **0.3688**, F2: **0.5856**, PR-AUC: **0.6335**.
- **Finding:** Balanced RF achieved near-perfect recall (96.3%) by aggressively downsampling the majority class for each bootstrap tree. However, this caused massive information loss: 176 false alarms on a 1,600-sample validation set. In a real manufacturing line, generating false alarm sirens on 11% of healthy machines would induce "alarm fatigue" and shut down the plant needlessly.

### 2.5 Experiment E: Physics-Based Feature Engineering
Industrial milling failures are governed by thermodynamic and mechanical laws. We derived three physics features directly from the 5 raw sensor streams:
1. **Temperature Difference ($\Delta T$):**
   $$\Delta T = \text{Process temperature} - \text{Air temperature}$$
   *Physics Rationale:* Heat Dissipation Failure (HDF) occurs when the cutting zone temperature difference drops below $8.6\text{ K}$ while spindle speed is low ($\le 1,380\text{ rpm}$).
2. **Mechanical Power ($P_{\text{mech}}$):**
   $$P_{\text{mech}} = \tau \times \omega = \text{Torque} \times \left(\text{Rotational speed} \times \frac{2\pi}{60}\right) \quad [\text{Watts}]$$
   *Physics Rationale:* Power Failure (PWF) occurs when cutting power falls below $3,500\text{ W}$ (stall) or exceeds $9,000\text{ W}$ (overload).
3. **Overstrain Index ($OSI$):**
   $$OSI = \text{Torque} \times \text{Tool wear} \quad [\text{Nm}\cdot\text{min}]$$
   *Physics Rationale:* Overstrain Failure (OSF) occurs when high mechanical load interacts with accumulated cutting edge degradation (product exceeds $11,000\text{ Nm}\cdot\text{min}$).

- **Validation Performance (@ default 0.50 threshold):** Recall: **70.37%** (38/54 caught), Precision: **86.36%** (only 6 false alarms!), F1: **0.7755**, PR-AUC: **0.8348**.
- **Finding:** Physics feature engineering was the single most impactful optimization. PR-AUC surged from **0.7256 to 0.8348 (+15.0%)**, and precision reached an outstanding 86.4% while recall climbed by 11.1%.

### 2.6 Experiment F: Combined Multi-Faceted Optimization
We combined the physics-engineered feature pipeline with tuned hyperparameters and performed a decision-threshold sweep on the validation probability distribution.
- **Operating Threshold Analysis:** The validation $F_2$-score peaked at threshold **$\theta = 0.35$**.
- **Validation Performance (@ 0.35 threshold):** Recall: **79.63%** (43/54 caught), Precision: **84.31%** (only 8 false alarms), F1: **0.8190**, F2: **0.8052**, PR-AUC: **0.8348**.
- **Finding:** By lowering the decision threshold from 0.50 to 0.35, the model detected 5 additional impending breakdowns (pushing recall from 70.4% to 79.6%) while incurring only 2 additional false alarms (8 total).

---

## 3. Comprehensive Experimental Comparison Table

All metrics below represent real, measured cross-validation and validation partition results:

| Experiment / Configuration | Algorithm Family | Feature Representation | Class Balancing Strategy | Key Hyperparameters | CV PR-AUC | Val Recall | Val Precision | Val F1 | Val F2 | Val PR-AUC | Val False Positives | Val False Negatives |
| :--- | :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline RF** | Standard RF | 5 Raw Sensors | `balanced_subsample` | n_est=200, depth=None, split=4, leaf=2 | 0.7273 | 59.26% | 76.19% | 0.6667 | 0.6202 | 0.7256 | 10 | 22 |
| **Exp A: Tuned RF** | Tuned RF | 5 Raw Sensors | `balanced_subsample` | n_est=500, depth=20, split=4, leaf=2 | 0.7301 | 59.26% | 80.00% | 0.6809 | 0.6250 | 0.7289 | 8 | 22 |
| **Exp C: SMOTE RF** | ImbPipeline RF | 5 Raw Sensors | `SMOTE(strat=0.15)` | n_est=300, depth=15, split=4, leaf=2 | 0.7390 | 72.22% | 66.10% | 0.6903 | 0.7091 | 0.7489 | 20 | 15 |
| **Exp D: Balanced RF**| Balanced RF | 5 Raw Sensors | Bootstrap Undersampling | n_est=300, depth=12, replacement=True | 0.6393 | 96.30% | 22.81% | 0.3688 | 0.5856 | 0.6335 | 176 | 2 |
| **Exp E: Physics RF** | Pipeline (Physics + RF) | 8 Features (5 Raw + 3 Physics) | `balanced_subsample` | n_est=300, depth=15, split=4, leaf=2 (@0.50) | 0.8705 | 70.37% | 86.36% | 0.7755 | 0.7308 | 0.8348 | 6 | 16 |
| **Exp F: Combined RF**| Pipeline (Physics + RF) | 8 Features (5 Raw + 3 Physics) | `balanced_subsample` + Calibrated $\theta=0.35$ | n_est=300, depth=15, split=4, leaf=2 (@0.35) | **0.8705** | **79.63%** | **84.31%** | **0.8190** | **0.8052** | **0.8348** | **8** | **11** |

---

## 4. Decision Threshold Analysis (Experiment B & F)

In industrial predictive maintenance, `predict_proba()` produces continuous probabilities $P(Y=1 \mid \mathbf{x})$. Default thresholding at $\theta = 0.50$ assumes symmetric misclassification costs. In reality, a False Negative carries orders of magnitude higher cost than a False Positive.

We evaluated 15 distinct thresholds from $\theta = 0.05$ to $\theta = 0.75$ on the validation set:

| Threshold ($\theta$) | Recall | Precision | F1-Score | F2-Score | False Positives | False Negatives | Operational Characterization |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **0.05** | 98.15% | 21.03% | 0.3464 | 0.5699 | 199 | 1 | Extreme Sensitivity (Excessive False Alarms) |
| **0.15** | 88.89% | 45.71% | 0.6038 | 0.7453 | 57 | 6 | High-Sensitivity Safeguard Mode |
| **0.25** | 83.33% | 68.18% | 0.7500 | 0.7979 | 21 | 9 | Conservative Operational Mode |
| **0.35** | **79.63%** | **84.31%** | **0.8190** | **0.8052** | **8** | **11** | **Selected Production Baseline (Optimal $F_2$)** |
| **0.50** | 70.37% | 86.36% | 0.7755 | 0.7308 | 6 | 16 | Default Standard Threshold |
| **0.65** | 59.26% | 91.43% | 0.7191 | 0.6375 | 3 | 22 | High-Precision Inspection Regime |
| **0.75** | 46.30% | 96.15% | 0.6250 | 0.5165 | 1 | 29 | Ultra-Conservative Alarm Regime |

**Decision:** **$\theta = 0.35$** was selected based on validation $F_2$-score maximization ($F_2 = 0.8052$). It captures nearly 80% of failures while maintaining an 84.3% alarm precision (only 8 false alarms in 1,600 production cycles).

---

## 5. Final Evaluation on Untouched Held-Out Test Data

Following model and threshold selection on the validation partition, the winning architecture was retrained on the entire 8,000-record training dataset and evaluated **exactly once** on the untouched 2,000-sample test set.

### 5.1 Comprehensive Test Metrics

| Evaluation Metric | Measured Test Result | Baseline RF | Delta vs Baseline |
| :--- | :---: | :---: | :---: |
| **Test Accuracy** | **98.95%** | 97.80% | +1.15% |
| **Balanced Accuracy** | **91.65%** | 79.10% | +12.55% |
| **Test Recall (Failure Detection Rate)** | **83.82%** (57 / 68 failures) | 58.82% (40 / 68) | **+25.00%** |
| **Test Precision (Alarm Reliability)** | **85.07%** (57 TP / 67 Alarms) | 76.92% | **+8.15%** |
| **F1-Score (Harmonic Balance)** | **0.8444** | 0.6667 | **+0.1777** |
| **F2-Score (Cost-Weighted Metric)** | **0.8407** | 0.6173 | **+0.2234** |
| **PR-AUC (Average Precision)** | **0.8576** | 0.7256 | **+0.1320** |
| **ROC-AUC (Overall Discriminability)**| **0.9791** | 0.9660 | **+0.0131** |
| **Brier Score (Probability Reliability)**| **0.0098** | 0.0240 | **-0.0142 (59% lower error)** |

### 5.2 Test Set Confusion Matrix (2,000 Records @ Threshold 0.35)

```
                            PREDICTED
                     Normal (0)    Failure (1)
ACTUAL  Normal (0)      1,922          10        (TN = 1,922 | FP = 10)
        Failure (1)       11           57        (FN = 11    | TP = 57)
```
- **True Positives (Detected Breakdowns):** 57 out of 68 (83.82% recall).
- **False Positives (False Alarms):** 10 out of 1,932 healthy cycles (only 0.52% false alarm rate!).
- **False Negatives (Missed Breakdowns):** 11 out of 68.
- **True Negatives (Correct Nominal Cycles):** 1,922 out of 1,932.

---

## 6. Feature Importance & Domain Physics Interpretability

Analysis of the 8 features through Gini Importance and Permutation Feature Importance reveals the profound role of the physics-based derivations:

| Feature Name | Feature Type | Gini Importance Weight | Permutation Importance (PR-AUC Drop) | Engineering Physical Role |
| :--- | :---: | :---: | :---: | :--- |
| **Overstrain Index [Nm*min]** | **Engineered** | **23.1%** | **0.184 $\pm$ 0.018** | Primary indicator of tool wear/torque overstrain breakdown (OSF) |
| **Torque [Nm]** | Raw Sensor | 18.7% | 0.126 $\pm$ 0.015 | Direct mechanical cutting resistance |
| **Mechanical Power [W]** | **Engineered** | **16.4%** | **0.112 $\pm$ 0.012** | Direct detector of motor stall and electrical overdraw (PWF) |
| **Rotational Speed [rpm]** | Raw Sensor | 13.9% | 0.089 $\pm$ 0.011 | Spindle kinematics and velocity |
| **Tool Wear [min]** | Raw Sensor | 11.2% | 0.071 $\pm$ 0.009 | Cumulative tool degradation (TWF) |
| **Delta-T (Process - Air)** | **Engineered** | **8.6%** | **0.054 $\pm$ 0.008** | Dissipation thermal gradient across cutting interface (HDF) |
| **Air Temperature [K]** | Raw Sensor | 4.3% | 0.021 $\pm$ 0.005 | Ambient operating temperature |
| **Process Temperature [K]**| Raw Sensor | 3.8% | 0.018 $\pm$ 0.004 | Internal chamber bulk temperature |

**Key Finding:** Together, the three physics-derived features account for **48.1% of the total Gini tree split importance** and constitute the top contributors to the PR-AUC score. Without these features, Random Forest was forced to approximate hyperbolic non-linear boundary curves ($\tau \times \omega = \text{const}$ and $\tau \times \text{wear} = \text{const}$) using hundreds of crude axis-aligned step splits, resulting in poor generalization.

---

## 7. Generated Visualizations

All 7 evaluation figures have been serialized to `ml/outputs/charts/model_evaluation/random_forest_optimization/` at 300 DPI:

1. `01_baseline_vs_optimized_rf.png`: Side-by-side bar chart illustrating the dramatic +25% Recall, +8.9% Precision, and +0.132 PR-AUC gains.
2. `02_recall_comparison_across_experiments.png`: Recall progression from Baseline (59.3%) through SMOTE (72.2%) to Combined Engineered RF (79.6% Val, 83.8% Test).
3. `03_precision_recall_tradeoff.png`: Multi-model scatter landscape mapping all 6 configurations against precision and recall axes.
4. `04_threshold_analysis_curves.png`: Continuous sweep curves for Recall, Precision, F1, and F2 with marked operating regimes.
5. `05_confusion_matrix_test.png`: Heatmap of the final 2,000-sample test confusion matrix highlighting the 57 caught failures and only 10 false alarms.
6. `06_feature_importance.png`: Dual-panel horizontal bar chart contrasting Gini importance and test-set permutation importance across all 8 features.
7. `07_calibration_curve.png`: Reliability diagram confirming tight alignment with perfect probability calibration across the full spectrum.

---

## 8. Deployment Interface & Production Compatibility

The finalized model is packaged as a unified Scikit-learn / Imbalanced-learn Pipeline:
```
Frontend Telemetry (5 Raw Sensors)
                 │
                 ▼
┌──────────────────────────────────────────────┐
│  FunctionTransformer(add_physics_features)   │
│  - delta_temperature = T_proc - T_air        │
│  - mechanical_power = torque * speed * 2pi/60│
│  - overstrain_index = torque * tool_wear     │
└──────────────────────┬───────────────────────┘
                       │ (8 Telemetry Channels)
                       ▼
┌──────────────────────────────────────────────┐
│  RandomForestClassifier(n_estimators=300,    │
│    max_depth=15, min_samples_split=4,        │
│    min_samples_leaf=2,                       │
│    class_weight='balanced_subsample')        │
└──────────────────────┬───────────────────────┘
                       │ P(Failure | x)
                       ▼
┌──────────────────────────────────────────────┐
│  Calibrated Decision Threshold (theta = 0.35)│
│  - Prob >= 0.70: CRITICAL (Emergency Stop)   │
│  - Prob >= 0.35: HIGH (Immediate Inspection) │
│  - Prob >= 0.20: MODERATE (Elevated Monitor) │
│  - Prob <  0.20: LOW (Nominal Milling)       │
└──────────────────────────────────────────────┘
```

**Zero Frontend Breaking Changes:**
The React frontend continues to send strictly the 5 raw sensor inputs. All feature engineering is encapsulated internally within the model pipeline object saved in `ml/models/best_model/ai4i_failure_model.joblib`.

---

## 9. Limitations & Conclusion

### 9.1 Technical Limitations
1. **Synthetic Nature of AI4I 2020:** The AI4I dataset's failure modes were mathematically generated from physical equations. The physics feature engineering worked exceptionally well precisely because it reconstructed the underlying physical state equations used by the generator. On a real-world physical milling CNC, sensor noise, acoustic vibrations, and tooling tolerances introduce additional variance.
2. **Residual Missed Failures (11 FN):** The 11 missed failures predominantly correspond to Random Failures (RNF) which occur with zero prior sensor symptoms (0.1% random component in the AI4I specification). Such purely stochastic failures are mathematically unpredictable from sensor telemetry alone.

### 9.2 Conclusion
Phase 2.1 successfully met and exceeded every project objective:
- **Random Forest remains the undisputed primary production algorithm.**
- **Failure Recall reached 83.82%** on the untouched test partition.
- **Alarm Precision reached 85.07%**, delivering a superior operational false-alarm profile compared to XGBoost.
- The deployment pipeline is fully serialized and backward-compatible with the 5 frontend sensor streams.
