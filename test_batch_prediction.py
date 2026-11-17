"""Tests for batch CSV scoring and bulk history storage.

Run from the project root:
    python -W ignore test_batch_prediction.py
"""

from __future__ import annotations

import io
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "ml" / "src"))

# Use a throwaway database BEFORE importing the storage module.
_TEMP_DIR = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
os.environ["MACHINEGUARD_DB_PATH"] = str(Path(_TEMP_DIR.name) / "batch_test.db")

from batch_prediction import (  # noqa: E402
    MAX_BATCH_ROWS,
    REPORT_COLUMNS,
    build_batch_report,
    map_columns,
    template_csv,
)
from database import (  # noqa: E402
    authenticate_user,
    create_user,
    get_predictions,
    init_db,
    save_predictions_bulk,
)
from model_prediction import EquipmentFailurePredictor  # noqa: E402

RAW_CSV = PROJECT_ROOT / "ml" / "data" / "raw" / "ai4i2020.csv"
PREDICTOR = EquipmentFailurePredictor()

SNAKE_CASE_ROWS = pd.DataFrame(
    {
        "machine_id": ["PRESS-001", "PRESS-002", "PRESS-003"],
        "air_temperature": [298.2, 301.5, 303.8],
        "process_temperature": [308.7, 310.8, 311.2],
        "rotational_speed": [1500, 1300, 1320],
        "torque": [40.0, 65.5, 52.0],
        "tool_wear": [35, 215, 140],
    }
)


def test_template_is_valid_and_scores() -> None:
    frame = pd.read_csv(io.StringIO(template_csv()))

    mapping, id_column, missing = map_columns(frame)
    assert id_column == "machine_id"
    assert missing == []

    report, summary = build_batch_report(frame, PREDICTOR)

    assert summary["total"] == 3
    assert summary["valid"] == 3
    assert summary["invalid"] == 0
    assert list(report.columns) == REPORT_COLUMNS
    assert set(report["machine_id"]) == {"PRESS-001", "PRESS-002", "PRESS-003"}


def test_500_row_ai4i_sample() -> None:
    frame = pd.read_csv(RAW_CSV, nrows=500)
    report, summary = build_batch_report(frame, PREDICTOR)

    assert summary["total"] == 500
    assert summary["valid"] == 500
    assert summary["invalid"] == 0

    # Row numbers account for the CSV header line.
    assert report["row_number"].min() == 2
    assert report["row_number"].max() == 501

    # Report is sorted from riskiest to safest.
    probabilities = report["failure_percentage"].tolist()
    assert probabilities == sorted(probabilities, reverse=True)

    assert summary["flagged"] <= summary["valid"]
    flagged = report[report["maintenance_required"] == "YES"]
    assert len(flagged) == summary["flagged"]
    assert set(flagged["prediction_code"]) == {1}

    counts = summary["risk_counts"]
    assert sum(counts.values()) == summary["valid"]
    assert counts["LOW"] + counts["MODERATE"] + counts["HIGH"] + counts["CRITICAL"] == 500


def test_snake_case_and_aliases() -> None:
    report, summary = build_batch_report(SNAKE_CASE_ROWS, PREDICTOR)

    assert summary["valid"] == 3
    assert report.iloc[0]["machine_id"] in {"PRESS-001", "PRESS-002", "PRESS-003"}

    # AI4I-style header names are also accepted.
    ai4i_style = SNAKE_CASE_ROWS.rename(
        columns={
            "air_temperature": "Air temperature [K]",
            "process_temperature": "Process temperature [K]",
            "rotational_speed": "Rotational speed [rpm]",
            "torque": "Torque [Nm]",
            "tool_wear": "Tool wear [min]",
            "machine_id": "Equipment ID",
        }
    )
    report2, summary2 = build_batch_report(ai4i_style, PREDICTOR)

    assert summary2["valid"] == 3
    assert set(report2["machine_id"]) == set(report["machine_id"])


def test_missing_required_column() -> None:
    frame = SNAKE_CASE_ROWS.drop(columns=["torque"])

    try:
        build_batch_report(frame, PREDICTOR)
    except ValueError as exc:
        message = str(exc)
        assert "Missing required column" in message
        assert "Torque" in message
    else:
        raise AssertionError("Expected ValueError for a missing torque column")


def test_invalid_rows_are_reported_not_fatal() -> None:
    frame = SNAKE_CASE_ROWS.copy()
    frame.loc[0, "torque"] = None                 # missing value
    frame.loc[1, "air_temperature"] = 999.0       # out of range
    frame.loc[2, "process_temperature"] = 200.0   # below air temperature

    report, summary = build_batch_report(frame, PREDICTOR)

    assert summary["total"] == 3
    assert summary["valid"] == 0
    assert summary["invalid"] == 3
    assert (report["status"] == "Invalid").all()
    assert (report["risk_level"] == "INVALID").all()
    assert (report["maintenance_required"] == "REVIEW").all()
    assert (report["prediction_code"] == -1).all()
    assert report["failure_percentage"].isna().all()
    assert report["error"].notna().all()

    # A good row still scores even when it sits beside invalid rows.
    mixed = pd.concat([frame, SNAKE_CASE_ROWS.iloc[[0]]], ignore_index=True)
    report2, summary2 = build_batch_report(mixed, PREDICTOR)

    assert summary2["valid"] == 1
    assert summary2["invalid"] == 3
    assert report2.iloc[0]["status"] == "OK"


def test_empty_and_oversized_files() -> None:
    for frame in (pd.DataFrame(), pd.DataFrame({"torque": []})):
        try:
            build_batch_report(frame, PREDICTOR)
        except ValueError as exc:
            assert "empty" in str(exc).lower()
        else:
            raise AssertionError("Expected ValueError for an empty upload")

    oversized = pd.DataFrame(
        {
            "air_temperature": [298.0] * (MAX_BATCH_ROWS + 1),
            "process_temperature": [308.0] * (MAX_BATCH_ROWS + 1),
            "rotational_speed": [1500] * (MAX_BATCH_ROWS + 1),
            "torque": [40.0] * (MAX_BATCH_ROWS + 1),
            "tool_wear": [10] * (MAX_BATCH_ROWS + 1),
        }
    )

    try:
        build_batch_report(oversized, PREDICTOR)
    except ValueError as exc:
        assert "limit" in str(exc).lower()
    else:
        raise AssertionError("Expected ValueError for too many rows")


def test_bulk_history_insert() -> None:
    init_db()
    user = create_user("Batch User", "batch@example.com", "batchpass123")
    assert authenticate_user("batch@example.com", "batchpass123")

    report, summary = build_batch_report(SNAKE_CASE_ROWS, PREDICTOR)
    valid = report[report["status"] == "OK"].head(3)

    records = [
        {
            "scenario": f"Batch upload · {row.machine_id}",
            "air_temperature": row.air_temperature,
            "process_temperature": row.process_temperature,
            "rotational_speed": row.rotational_speed,
            "torque": row.torque,
            "tool_wear": row.tool_wear,
            "failure_probability": row.failure_probability,
            "prediction_code": int(row.prediction_code),
            "risk_level": row.risk_level,
            "decision_threshold": PREDICTOR.threshold,
            "recommended_action": row.recommended_action,
            "model_name": PREDICTOR.model_name,
        }
        for row in valid.itertuples()
    ]

    assert save_predictions_bulk(user["id"], []) == 0
    assert save_predictions_bulk(user["id"], records) == 3

    stored = get_predictions(user["id"])
    assert len(stored) == 3
    assert {row["scenario"] for row in stored} == {
        record["scenario"] for record in records
    }
    assert all(row["risk_level"] != "INVALID" for row in stored)


if __name__ == "__main__":
    failures = 0
    for name, case in sorted(globals().items()):
        if name.startswith("test_") and callable(case):
            try:
                case()
                print(f"PASS {name}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"FAIL {name}: {exc}")
    _TEMP_DIR.cleanup()
    raise SystemExit(1 if failures else 0)
