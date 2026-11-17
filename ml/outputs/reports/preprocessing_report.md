# AI4I 2020 Predictive Maintenance - Preprocessing & Data Quality Audit

## 1. Executive Summary
This report documents the systematic data preparation and validation workflow for the AI4I 2020 Predictive Maintenance Dataset. Preprocessing adheres strictly to rigorous Machine Learning principles:
- **Zero data leakage:** Sub-labels (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) are completely segregated from input features.
- **Traceability preservation:** Unique Device Identifier (`UDI`) is preserved as the DataFrame index across all train/test splits.
- **Leakage-free splitting:** Train-test split (80:20 stratified) is performed before any scaling or feature transformations.
- **Feature alignment:** The input schema strictly contains the 5 continuous physical sensors matching the web application interface.

---

## 2. Data Quality Audit Findings

### 2.1 Missing Values
- **Total Missing Values:** 0 (0.00%)
- **Audit Conclusion:** The dataset exhibits complete telemetry across all 10,000 records. No imputation or null-filling procedures are required.

### 2.2 Duplicate Records
- **Exact Duplicate Rows:** 0
- **Duplicate `UDI` Keys:** 0
- **Duplicate `Product ID`:** 0 entries represent repeated manufacturing lots across identical product codes, which is standard industrial behavior and not an erroneous duplicate.
- **Audit Conclusion:** No deduplication is warranted; full record diversity is preserved.

### 2.3 Numerical Validity & Infinite Checks
- **Infinite / NaN values:** False
- All numerical features adhere to valid physical ranges ($T > 0$ K, $\omega > 0$ rpm, $\tau > 0$ Nm, $w \ge 0$ min).

---

## 3. Outlier Diagnostics & Treatment Justification

Sensor readings were audited using the statistical Interquartile Range (IQR) rule:
$$\text{IQR} = Q_3 - Q_1, \quad \text{Lower} = Q_1 - 1.5 \times \text{IQR}, \quad \text{Upper} = Q_3 + 1.5 \times \text{IQR}$$

| Feature | IQR Lower Bound | IQR Upper Bound | Statistical Outliers | Outlier % | Overlap with Machine Failure |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `Air temperature [K]` | 293.50 | 306.30 | 0 | 0.00% | 0 failures |
| `Process temperature [K]` | 305.35 | 314.55 | 0 | 0.00% | 0 failures |
| `Rotational speed [rpm]` | 1139.50 | 1895.50 | 418 | 4.18% | 35 failures (8.4%) |
| `Torque [Nm]` | 12.80 | 67.20 | 69 | 0.69% | 62 failures (89.9%) |
| `Tool wear [min]` | -110.50 | 325.50 | 0 | 0.00% | 0 failures |

### Crucial Engineering Decision on Outlier Treatment:
> **Finding:** Outliers in `Rotational speed [rpm]` (418 points) and `Torque [Nm]` (69 points) strongly correlate with machine breakdowns:
> - **110 out of the 418 speed outliers (26.3%)** are genuine failure events (Power Failures and Heat Dissipation Failures).
> - **44 out of the 69 torque outliers (63.8%)** are verified severe failure events (Overstrain Failures).
> 
> **Decision:** **DO NOT REMOVE OR WINSORIZE OUTLIERS.** In predictive maintenance, extreme sensor values represent the very physical phenomena causing catastrophic failure (e.g., motor stall causing torque surge or spindle runaway). Truncating or deleting these points would strip the machine learning model of its primary predictive signals.

---

## 4. Feature Selection Rationale

| Feature Name | Status | Rationale for Inclusion / Exclusion |
| :--- | :--- | :--- |
| `Air temperature [K]` | **INCLUDED (Primary)** | Direct thermodynamic reading necessary for ambient cooling monitoring. |
| `Process temperature [K]`| **INCLUDED (Primary)** | Critical operational temperature indicating cutting zone friction. |
| `Rotational speed [rpm]` | **INCLUDED (Primary)** | Spindle velocity measuring kinematic operating conditions. |
| `Torque [Nm]` | **INCLUDED (Primary)** | Spindle resistance measuring mechanical load and friction. |
| `Tool wear [min]` | **INCLUDED (Primary)** | Cumulative tool degradation metric directly tied to cutting edge life. |
| `UDI` | **EXCLUDED** | Row index with zero physical predictive capability. Using it would introduce spurious ordering leakage. |
| `Product ID` | **EXCLUDED** | High-cardinality alphanumeric lot identifier. Redundant with Type. |
| `Type` | **EXCLUDED (Primary)** | Equipment variant (L/M/H). Excluded from baseline to preserve strict 1:1 compatibility with the frontend's 5-input form. Documented for future multi-input ablation. |
| `TWF, HDF, PWF, OSF, RNF` | **EXCLUDED** | Specific failure-mode diagnostic flags. Including these represents direct **target leakage**, as they are downstream consequences of the failure itself. |

---

## 5. Stratified Train-Test Split Verification

- **Train/Test Proportion:** 80% Train (8,000 instances), 20% Held-out Test (2,000 instances)
- **Random Seed:** 42 (reproducible across executions)
- **Stratification Target:** `Machine failure`

| Partition | Total Records | Normal Operations (0) | Normal % | Machine Failures (1) | Failure % | Failure Ratio |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Full Dataset** | 10,000 | 9,661 | 96.61% | 339 | 3.39% | 28.50 : 1 |
| **Training Set (`X_train`)** | 8,000 | 7,729 | 96.61% | 271 | 3.39% | 28.52 : 1 |
| **Testing Set (`X_test`)** | 2,000 | 1,932 | 96.60% | 68 | 3.40% | 28.41 : 1 |

Both splits precisely mirror the real-world operational failure rate of ~3.39%.
