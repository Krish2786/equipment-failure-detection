"""
Batch CSV scoring for MachineGuard
==================================

Turns an uploaded spreadsheet of machine telemetry into a per-machine risk
report. The report is sorted by failure probability so the machines that need
maintenance appear first.

Accepted column names are deliberately flexible:

    air_temperature / Air temperature [K] / airTemperature / Air Temperature
    process_temperature / Process temperature [K] / processTemperature
    rotational_speed / Rotational speed [rpm] / rotationalSpeed
    torque / Torque [Nm]
    tool_wear / Tool wear [min] / toolWear
    machine_id / UDI / Equipment ID / Asset ID   (optional)

The five sensor values are required. Extra columns are ignored.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from model_prediction import diagnose_failure_mode

MAX_BATCH_ROWS = 5000

SENSOR_FIELDS = [
    "air_temperature",
    "process_temperature",
    "rotational_speed",
    "torque",
    "tool_wear",
]

SENSOR_LABELS = {
    "air_temperature": "Air temperature [K]",
    "process_temperature": "Process temperature [K]",
    "rotational_speed": "Rotational speed [rpm]",
    "torque": "Torque [Nm]",
    "tool_wear": "Tool wear [min]",
}

MODEL_COLUMNS = {
    "air_temperature": "Air temperature [K]",
    "process_temperature": "Process temperature [K]",
    "rotational_speed": "Rotational speed [rpm]",
    "torque": "Torque [Nm]",
    "tool_wear": "Tool wear [min]",
}

# Same broad sanity bounds as the single-prediction form in app.py.
BOUNDS = {
    "air_temperature": (250.0, 350.0),
    "process_temperature": (250.0, 400.0),
    "rotational_speed": (0.0, 10000.0),
    "torque": (0.0, 200.0),
    "tool_wear": (0.0, 1000.0),
}

SENSOR_ALIASES = {
    "air_temperature": [
        "air temperature k", "air temperature", "airtemperature",
        "air temp", "airtemp", "ambient temperature",
    ],
    "process_temperature": [
        "process temperature k", "process temperature", "processtemperature",
        "process temp", "processtemp", "cutting temperature",
    ],
    "rotational_speed": [
        "rotational speed rpm", "rotational speed", "rotationalspeed",
        "rotational speed r min", "speed", "rpm",
    ],
    "torque": [
        "torque nm", "torque", "torque n m", "torque newton meters",
    ],
    "tool_wear": [
        "tool wear min", "tool wear", "toolwear", "tool wear minutes",
        "tool wear minute", "wear",
    ],
}

ID_ALIASES = [
    "udi", "machine id", "machine identifier", "equipment id", "asset id",
    "machine", "equipment", "asset", "product id", "id", "name",
]

REPORT_COLUMNS = [
    "machine_id",
    "row_number",
    "status",
    "air_temperature",
    "process_temperature",
    "rotational_speed",
    "torque",
    "tool_wear",
    "failure_probability",
    "failure_percentage",
    "prediction_code",
    "risk_level",
    "maintenance_required",
    "recommended_action",
    "failure_mode",
    "failure_mode_reason",
    "error",
]


def normalize_column_name(name: Any) -> str:
    """Lower-case and strip punctuation so aliases can be matched."""
    text = str(name).replace("\ufeff", " ")
    text = text.strip().lower()
    for char in "_-[]()/":
        text = text.replace(char, " ")
    return " ".join(text.split())


def map_columns(df: pd.DataFrame) -> tuple[dict[str, str], str | None, list[str]]:
    """Return {field: source_column}, the machine-id column, and missing fields."""
    normalized = {normalize_column_name(column): column for column in df.columns}

    mapping: dict[str, str] = {}
    missing: list[str] = []

    for field, aliases in SENSOR_ALIASES.items():
        source = next(
            (normalized[alias] for alias in aliases if alias in normalized),
            None,
        )
        if source is None:
            missing.append(field)
        else:
            mapping[field] = source

    id_column = next(
        (normalized[alias] for alias in ID_ALIASES if alias in normalized),
        None,
    )
    return mapping, id_column, missing


def classify_probability(probability: float, threshold: float) -> tuple[int, str, str]:
    """Mirror EquipmentFailurePredictor.predict's risk tiers and actions."""
    if probability >= 0.70:
        return (
            1,
            "CRITICAL",
            "Immediate spindle shutdown and comprehensive mechanical inspection required.",
        )
    if probability >= threshold:
        return (
            1,
            "HIGH",
            "Schedule preventive maintenance inspection before next production batch.",
        )
    if probability >= 0.20:
        return (
            0,
            "MODERATE",
            "Telemetry elevated; increase sensor monitoring frequency.",
        )
    return (
        0,
        "LOW",
        "Standard operational parameters; proceed with scheduled milling.",
    )


def _validate_row(row: pd.Series) -> str | None:
    problems: list[str] = []

    for field in SENSOR_FIELDS:
        value = row[field]
        low, high = BOUNDS[field]
        if pd.isna(value):
            problems.append(f"{SENSOR_LABELS[field]} is missing or non-numeric")
        elif value < low or value > high:
            problems.append(
                f"{SENSOR_LABELS[field]} {value:g} is outside "
                f"the accepted range {low:g} to {high:g}"
            )

    if not problems and row["process_temperature"] < row["air_temperature"]:
        problems.append("Process temperature is below air temperature")

    return "; ".join(problems) if problems else None


def build_batch_report(
    df: pd.DataFrame,
    predictor,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Score every row and return (report_dataframe, summary_dict)."""
    if df is None or df.empty:
        raise ValueError("The uploaded file is empty.")

    if len(df) > MAX_BATCH_ROWS:
        raise ValueError(
            f"The file has {len(df):,} rows; the limit is {MAX_BATCH_ROWS:,}. "
            "Please split it into smaller files."
        )

    mapping, id_column, missing = map_columns(df)

    if missing:
        missing_labels = ", ".join(SENSOR_LABELS[field] for field in missing)
        detected = ", ".join(str(column) for column in df.columns)
        raise ValueError(
            f"Missing required column(s): {missing_labels}. "
            f"Detected columns: {detected}. "
            "Download the template for the expected layout."
        )

    working = pd.DataFrame(index=df.index)
    working["row_number"] = np.arange(2, len(df) + 2)  # CSV header is row 1

    if id_column is None:
        working["machine_id"] = [f"ROW-{i}" for i in range(1, len(df) + 1)]
    else:
        raw_ids = df[id_column].astype("string").str.strip()
        working["machine_id"] = [
            value
            if value is not pd.NA and str(value).lower() not in {"", "nan", "none", "<na>"}
            else f"ROW-{index}"
            for index, value in enumerate(raw_ids, start=1)
        ]

    for field, source in mapping.items():
        working[field] = pd.to_numeric(df[source], errors="coerce")

    errors = []
    for _, row in working.iterrows():
        errors.append(_validate_row(row))

    working["error"] = errors
    working["status"] = ["Invalid" if error else "OK" for error in errors]

    working["failure_probability"] = np.nan
    working["failure_percentage"] = np.nan
    working["prediction_code"] = -1
    working["risk_level"] = "INVALID"
    working["maintenance_required"] = "REVIEW"
    working["recommended_action"] = ""
    working["failure_mode"] = "Not applicable"
    working["failure_mode_reason"] = ""

    valid_rows = working[working["status"] == "OK"]

    if len(valid_rows) > 0:
        model_input = pd.DataFrame(
            {
                MODEL_COLUMNS[field]: valid_rows[field].astype(float)
                for field in SENSOR_FIELDS
            }
        )[predictor.features]

        probabilities = predictor.model.predict_proba(model_input)[:, 1]
        threshold = float(predictor.threshold)

        for index, probability in zip(valid_rows.index, probabilities):
            probability = float(probability)
            code, risk_level, action = classify_probability(probability, threshold)

            working.at[index, "failure_probability"] = probability
            working.at[index, "failure_percentage"] = round(probability * 100, 2)
            working.at[index, "prediction_code"] = code
            working.at[index, "risk_level"] = risk_level
            working.at[index, "maintenance_required"] = "YES" if code == 1 else "NO"
            working.at[index, "recommended_action"] = action
            mode = diagnose_failure_mode(
                valid_rows.at[index, "air_temperature"],
                valid_rows.at[index, "process_temperature"],
                valid_rows.at[index, "rotational_speed"],
                valid_rows.at[index, "torque"],
                valid_rows.at[index, "tool_wear"],
            )
            working.at[index, "failure_mode"] = mode["mode_label"]
            working.at[index, "failure_mode_reason"] = mode["reason"]

    working = working.sort_values(
        by=["failure_percentage", "row_number"],
        ascending=[False, True],
        na_position="last",
    ).reset_index(drop=True)

    report = working[REPORT_COLUMNS].copy()

    valid_mask = report["status"] == "OK"
    flagged_mask = report["maintenance_required"] == "YES"
    valid_count = int(valid_mask.sum())
    flagged_count = int(flagged_mask.sum())

    risk_counts = (
        report.loc[valid_mask, "risk_level"]
        .value_counts()
        .reindex(["LOW", "MODERATE", "HIGH", "CRITICAL"])
        .fillna(0)
        .astype(int)
        .to_dict()
    )

    top_machine = None
    if valid_count:
        top_machine = str(
            report.loc[valid_mask]
            .sort_values("failure_percentage", ascending=False)
            .iloc[0]["machine_id"]
        )

    summary = {
        "total": int(len(report)),
        "valid": valid_count,
        "invalid": int(len(report) - valid_count),
        "flagged": flagged_count,
        "flagged_pct": round(flagged_count / valid_count * 100, 1) if valid_count else 0.0,
        "avg_percentage": (
            round(float(report.loc[valid_mask, "failure_percentage"].mean()), 2)
            if valid_count
            else 0.0
        ),
        "max_percentage": (
            round(float(report.loc[valid_mask, "failure_percentage"].max()), 2)
            if valid_count
            else 0.0
        ),
        "top_machine": top_machine,
        "risk_counts": risk_counts,
    }

    return report, summary


def template_csv() -> str:
    """A ready-to-edit CSV with one healthy and two risky examples."""
    return (
        "machine_id,air_temperature,process_temperature,"
        "rotational_speed,torque,tool_wear\n"
        "PRESS-001,298.2,308.7,1500,40.0,35\n"
        "PRESS-002,301.5,310.8,1300,65.5,215\n"
        "PRESS-003,303.8,311.2,1320,52.0,140\n"
    )
