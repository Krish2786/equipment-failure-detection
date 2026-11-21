# MachineGuard — Equipment Failure Detection System

> **Infosys Springboard Virtual Internship 7.0 (Batch 4 — AI Domain)**  
> **Project:** Machine Learning System for Early Detection of Equipment Failure  
> **Dataset:** AI4I 2020 Predictive Maintenance Dataset (10,000 continuous CNC milling telemetry records)

---

## 📌 Project Overview

**MachineGuard** is an industrial predictive maintenance system designed to detect impending machinery breakdowns before catastrophic failure occurs. Missing a mechanical failure in automated manufacturing can lead to spindle seizure, shattered cutting tools, costly production halts, and safety hazards.

Developed as part of the **Infosys Springboard Virtual Internship 7.0 (Batch 4 - AI Domain)**, MachineGuard provides an end-to-end, real-time diagnostic platform. It continuously processes raw telemetry from 5 physical milling machine sensors, enriches the signals with domain-derived thermodynamic and mechanical physics features, and feeds them into an optimized **scikit-learn Random Forest** pipeline.

The user-facing solution is delivered as an interactive **Streamlit** application (`app.py`), backed by a secure **SQLite** storage engine (`database.py`) and a high-throughput **batch CSV processing engine** (`batch_prediction.py`).

---

## 🚀 Key Features

- **PBKDF2-HMAC-SHA256 Authentication:**
  - Secure local account creation and login implemented using Python's standard library (`hashlib`, `hmac`, `secrets`).
  - Cryptographic protection with unique random salts, 260,000 hash iterations, and constant-time string comparison (`hmac.compare_digest`) to resist timing attacks.
  - Zero plaintext password storage.

- **Real-Time Single Telemetry Prediction:**
  - Instant failure risk scoring based on 5 continuous sensor inputs:
    - Air temperature [K]
    - Process temperature [K]
    - Rotational speed [rpm]
    - Torque [Nm]
    - Tool wear [min]
  - Visual probability gauges, classified risk levels (**LOW**, **MEDIUM**, **HIGH**), and quick-load operational scenario presets (Normal Operation, Overstrain, Heat Dissipation, Power Failure, Tool Wear).

- **High-Throughput Batch CSV Scoring (Up to 5,000 Rows):**
  - Drag-and-drop CSV ingestion capable of scoring entire equipment fleets (up to 5,000 machines per upload).
  - Flexible column header resolution supporting standard names, snake_case, camelCase, and AI4I bracketed formats (e.g., `Air temperature [K]`, `Rotational speed [rpm]`).
  - Robust row validation that flags invalid rows without aborting the batch run.
  - Dual report exports: **Complete fleet assessment report** and **prioritized maintenance shortlist** sorted from highest to lowest failure risk.

- **Physics-Informed Feature Engineering:**
  - Transparent mathematical signal enrichment executed inside an automated scikit-learn `FunctionTransformer` pipeline:
    - **Temperature Difference ($\Delta T$):** $\text{Process Temperature} - \text{Air Temperature}$
    - **Mechanical Power ($P$):** $2\pi \times \text{Rotational Speed} \times \text{Torque} / 60$
    - **Overstrain Index ($OSI$):** $\text{Torque} \times \text{Tool Wear}$

- **Physics-Based Failure Mode Diagnostics:**
  - Derives physics-based failure mechanism flags alongside statistical probability:
    - **HDF:** Heat Dissipation Failure
    - **PWF:** Power Failure
    - **OSF:** Overstrain Failure
    - **TWF:** Tool Wear Failure
    - **RNF:** Random Failure

- **Persistent SQLite Prediction History & Analytics:**
  - Automatically records all single and batch predictions into `data/machineguard.db` linked to the active user account.
  - Interactive history explorer featuring metric summary cards, search and filter controls, detailed breakdown views, side-by-side comparison of historical predictions, and CSV history exports.

- **Asset ID Tracking & Degradation Trend Charts:**
  - Track individual machinery assets over time with customizable Machine / Asset IDs (e.g., `MILL-01`, `CNC-04`, `PRESS-001`).
  - Interactive dual-axis degradation charts (powered by Plotly) displaying chronological failure probability escalation against the 35% operational decision threshold alongside cutting tool wear progression.

- **Analytical Fleet Overview Dashboard:**
  - High-level bird's-eye view of all tracked machinery assets across the industrial facility.
  - Real-time fleet KPI telemetry: active machine count, assets in HIGH/CRITICAL state, count above threshold, and fleet average risk.
  - Interactive machine asset cards equipped with live risk badges (`LOW`, `MODERATE`, `HIGH`, `CRITICAL`), trend trajectory indicators (📈 worsening, 📉 improving, ➡️ stable), failure rates, and last check timestamps.
  - Fleet-wide comparative analytics: risk comparison bar charts, tool wear distribution, fleet risk level breakdown pie charts, and tool wear vs. failure risk bubble scatter plots.
  - Deep-dive asset inspection view and sortable fleet summary data table.

---

## 📊 Model Architecture & Performance

The core prediction model is a scikit-learn pipeline composed of a domain-physics `FunctionTransformer` and an optimized `RandomForestClassifier`.

Because industrial failures represent only **3.39%** of operational records in the AI4I dataset (a class imbalance ratio of **28.5 : 1**), the default classification threshold of `0.50` produces excessive false negatives. To safeguard critical industrial machinery, the operational decision threshold was calibrated to **`0.35`** with balanced subsampling.

### Final Test Evaluation Metrics (Held-out Test Set: 2,000 Records)

| Metric | Value |
|---|---|
| **Model Type** | Optimized Random Forest (`Pipeline(FunctionTransformer + RandomForestClassifier)`) |
| **Accuracy** | **98.95%** |
| **Failure Detection Recall** | **83.82%** (57 of 68 failures captured) |
| **Precision** | **85.07%** |
| **F1-Score** | **84.44%** |
| **F2-Score (Recall-Weighted)** | **84.07%** |
| **Balanced Accuracy** | **91.65%** |
| **ROC-AUC** | **0.9791** |
| **PR-AUC** | **0.8576** |
| **Operational Decision Threshold** | **0.35** |

### Confusion Matrix on Test Set
- **True Positives (Correctly Detected Failures):** 57
- **False Negatives (Missed Failures):** 11
- **False Positives (False Alarms):** 10
- **True Negatives (Normal Operations):** 1,922

---

## 📁 Project Structure

```text
equipment-failure-detection/
├── app.py                          # Streamlit application & interactive dashboard
├── database.py                     # SQLite database management & PBKDF2 authentication
├── batch_prediction.py             # Batch CSV validation, scoring & report generation
├── requirements.txt                # Python package dependencies
├── LICENSE                         # MIT License
├── README.md                       # Comprehensive project documentation
├── ai4i2020.csv                    # AI4I 2020 Predictive Maintenance Dataset
├── test_database.py                # Unit tests for SQLite database and authentication
├── test_batch_prediction.py        # Unit tests for batch CSV processing & scoring
├── test_app_auth.py                # End-to-end integration tests for UI and workflows
│
├── ml/                             # Machine learning development pipeline
│   ├── data/
│   │   ├── raw/                    # Raw AI4I 2020 dataset
│   │   ├── processed/              # Cleaned telemetry dataset
│   │   └── splits/                 # Stratified train/test splits (80/20)
│   ├── models/
│   │   ├── best_model/             # Serialized production model & metadata
│   │   │   ├── ai4i_failure_model.joblib
│   │   │   └── model_metadata.json
│   │   └── candidates/             # Candidate benchmark models (DT, XGBoost, etc.)
│   ├── notebooks/                  # Jupyter notebooks for EDA and experimentation
│   ├── outputs/                    # Confusion matrices, ROC curves, and reports
│   └── src/
│       ├── model_prediction.py     # Production inference and diagnostics interface
│       ├── optimize_random_forest.py # Training, tuning, and threshold calibration
│       ├── preprocess_data.py      # Data cleaning and feature engineering
│       ├── evaluate_models.py      # Candidate model benchmarking
│       └── utils.py                # Shared ML utility functions
│
└── data/                           # Local database storage (excluded from git)
    └── machineguard.db             # User accounts and prediction history SQLite DB
```

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.10, 3.11, or 3.12
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/Krish2786/equipment-failure-detection.git
cd equipment-failure-detection
```

### 2. Set Up a Virtual Environment
```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application
```bash
streamlit run app.py
```
Open your browser and navigate to `http://localhost:8501`.

### 5. Run the Test Suite (Optional)
```bash
python test_database.py
python test_batch_prediction.py
python test_app_auth.py
```

---

## 🔒 Authentication & Data Storage

- On the initial run, users create a local account with **Full Name**, **Email**, and a password (minimum 8 characters).
- All passwords are hashed using **PBKDF2-HMAC-SHA256** with 260,000 iterations and salt generation via `secrets.token_hex(16)`.
- User profiles and timestamped telemetry predictions are stored locally in `data/machineguard.db`.
- The database file is automatically excluded from version control to protect user data and operational logs.

---

## 📜 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for full details.

---

## 👥 Acknowledgments

- **Infosys Springboard Virtual Internship 7.0 (Batch 4 - AI Domain)** for mentorship, project guidelines, and technical support.
- **AI4I 2020 Predictive Maintenance Dataset** provided by Stephan Matzka at the UCI Machine Learning Repository.
