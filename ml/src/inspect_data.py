"""
AI4I 2020 Predictive Maintenance Dataset - Data Inspection Script
==================================================================
Inspects dataset structure, dimensions, data types, missing values,
duplicates, distributions, and column semantics without modifying raw data.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

COLUMN_DESCRIPTIONS = {
    "UDI": "Unique Device Identifier (integer 1 to 10,000) identifying each record.",
    "Product ID": "Variant identifier consisting of letter (L/M/H) and product serial number.",
    "Type": "Equipment quality variant: L (Low, ~50%), M (Medium, ~30%), H (High, ~20%).",
    "Air temperature [K]": "Ambient air temperature in Kelvin (generated via random walk around 300 K).",
    "Process temperature [K]": "Manufacturing process temperature in Kelvin (~10 K above air temperature).",
    "Rotational speed [rpm]": "Spindle rotational speed calculated from ~2860 W power with noise overlay.",
    "Torque [Nm]": "Torque exerted by the tool, normally distributed around 40 Nm with ~10 Nm std dev.",
    "Tool wear [min]": "Cumulative tool wear time in minutes (increases by 2/3/5 min per workpiece for L/M/H).",
    "Machine failure": "Primary target variable (0 = Normal operational state, 1 = Machine failure).",
    "TWF": "Tool Wear Failure mode flag (tool replaced or failed due to wear beyond threshold).",
    "HDF": "Heat Dissipation Failure mode flag (insufficient cooling delta-T at low spindle speeds).",
    "PWF": "Power Failure mode flag (mechanical power outside safe operating limits: <3500W or >9000W).",
    "OSF": "Overstrain Failure mode flag (excessive product of tool wear and torque based on variant).",
    "RNF": "Random Failure mode flag (stochastic failure with ~0.1% occurrence probability)."
}

def find_raw_dataset() -> Path:
    """Finds raw CSV either in ml/data/raw or in workspace root."""
    candidate_paths = [
        Path(__file__).resolve().parent.parent / "data" / "raw" / "ai4i2020.csv",
        Path(__file__).resolve().parent.parent.parent / "ai4i2020.csv",
        Path("ml/data/raw/ai4i2020.csv"),
        Path("ai4i2020.csv")
    ]
    for p in candidate_paths:
        if p.exists() and p.is_file():
            return p
    raise FileNotFoundError(
        "Could not locate 'ai4i2020.csv' in ml/data/raw/ or workspace root. "
        "Please ensure the file exists."
    )

def inspect_dataset(df: pd.DataFrame) -> dict:
    """Performs comprehensive data inspection on the AI4I dataset."""
    results = {}
    
    # 1. Shape
    results["shape"] = df.shape
    results["n_rows"], results["n_cols"] = df.shape
    
    # 2. Columns & Data Types
    results["columns"] = list(df.columns)
    results["dtypes"] = {col: str(dtype) for col, dtype in df.dtypes.items()}
    
    # 3. Head & Tail
    results["head"] = df.head(5)
    results["tail"] = df.tail(5)
    
    # 4. Statistical Summary
    results["describe_num"] = df.describe().T
    
    # 5. Missing Values
    missing_count = df.isnull().sum()
    missing_pct = (missing_count / len(df)) * 100.0
    results["missing"] = pd.DataFrame({
        "Missing Count": missing_count,
        "Percentage (%)": missing_pct
    })
    
    # 6. Duplicates
    results["exact_duplicates"] = int(df.duplicated().sum())
    results["duplicate_udi"] = int(df["UDI"].duplicated().sum()) if "UDI" in df.columns else 0
    results["duplicate_product_id"] = int(df["Product ID"].duplicated().sum()) if "Product ID" in df.columns else 0
    
    # 7. Categorical Cardinality
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    results["cat_unique"] = {col: df[col].nunique() for col in cat_cols}
    results["cat_distributions"] = {col: df[col].value_counts().to_dict() for col in cat_cols}
    
    # 8. Target Class Distribution
    if "Machine failure" in df.columns:
        counts = df["Machine failure"].value_counts()
        n_normal = int(counts.get(0, 0))
        n_failure = int(counts.get(1, 0))
        total = len(df)
        results["target_distribution"] = {
            "normal_count": n_normal,
            "failure_count": n_failure,
            "normal_pct": (n_normal / total) * 100.0,
            "failure_pct": (n_failure / total) * 100.0,
            "imbalance_ratio": (n_normal / n_failure) if n_failure > 0 else np.nan
        }
        
    # 9. Failure Modes Breakdown
    failure_mode_cols = ["TWF", "HDF", "PWF", "OSF", "RNF"]
    existing_fm_cols = [c for c in failure_mode_cols if c in df.columns]
    results["failure_modes"] = {c: int(df[c].sum()) for c in existing_fm_cols}
    
    # Check discrepancy between Machine failure and failure mode sum
    if existing_fm_cols and "Machine failure" in df.columns:
        fm_any = (df[existing_fm_cols].sum(axis=1) > 0).astype(int)
        results["any_failure_mode_flagged"] = int(fm_any.sum())
        results["mf_1_no_modes"] = int(((df["Machine failure"] == 1) & (fm_any == 0)).sum())
        results["modes_1_mf_0"] = int(((df["Machine failure"] == 0) & (fm_any == 1)).sum())
        
    return results

def print_inspection_summary(results: dict):
    """Prints a clear, formatted inspection report to stdout."""
    print("=" * 80)
    print("AI4I 2020 PREDICTIVE MAINTENANCE DATASET - INSPECTION SUMMARY")
    print("=" * 80)
    print(f"Dataset Shape: {results['n_rows']} rows x {results['n_cols']} columns")
    print("\n--- Column List & Inferred Types ---")
    for col in results["columns"]:
        print(f"  • {col:<26}: {results['dtypes'][col]:<10} | {COLUMN_DESCRIPTIONS.get(col, 'N/A')}")
        
    print("\n--- Data Quality Checks ---")
    print(f"Total exact duplicate rows: {results['exact_duplicates']}")
    print(f"Duplicate UDI entries:     {results['duplicate_udi']}")
    print(f"Duplicate Product IDs:     {results['duplicate_product_id']} (Expected: multiple runs per product variant/batch)")
    
    print("\nMissing Values per Column:")
    missing_df = results["missing"]
    if missing_df["Missing Count"].sum() == 0:
        print("  Zero missing values detected across all 14 columns.")
    else:
        for col, row in missing_df.iterrows():
            if row["Missing Count"] > 0:
                print(f"  • {col}: {row['Missing Count']} ({row['Percentage (%)']:.2f}%)")
                
    print("\n--- Categorical Variables ---")
    for col, unq in results["cat_unique"].items():
        print(f"  • {col}: {unq} unique values")
        print(f"    Value counts: {results['cat_distributions'][col]}")
        
    print("\n--- Target Variable ('Machine failure') ---")
    td = results.get("target_distribution", {})
    print(f"  • Normal Operational (Class 0): {td.get('normal_count')} ({td.get('normal_pct', 0):.2f}%)")
    print(f"  • Machine Failure    (Class 1): {td.get('failure_count')} ({td.get('failure_pct', 0):.2f}%)")
    print(f"  • Failure-to-Normal Ratio:      1 : {td.get('imbalance_ratio', 0):.2f}")
    
    print("\n--- Failure Modes Breakdown ---")
    for fm, count in results.get("failure_modes", {}).items():
        print(f"  • {fm:<4} ({COLUMN_DESCRIPTIONS.get(fm, '').split('(')[0].strip()}): {count} occurrences")
    print(f"  • Any failure mode flagged:     {results.get('any_failure_mode_flagged')}")
    print(f"  • Machine failure=1 with no modes: {results.get('mf_1_no_modes')} (unexplained failure cases)")
    print(f"  • Mode flagged with Machine failure=0: {results.get('modes_1_mf_0')} (specifically RNF without causing machine shutdown)")
    print("=" * 80)

def main():
    try:
        csv_path = find_raw_dataset()
        print(f"Loading raw dataset from: {csv_path}")
        df = pd.read_csv(csv_path)
        results = inspect_dataset(df)
        print_inspection_summary(results)
    except Exception as e:
        print(f"Error during dataset inspection: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
