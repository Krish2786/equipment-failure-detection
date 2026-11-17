"""Tests for the SQLite user + prediction-history layer.

Run from the project root:
    python -W ignore test_database.py
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
from pathlib import Path

# Point the module at a throwaway database BEFORE importing it.
_TEMP_DIR = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
_DB_PATH = Path(_TEMP_DIR.name) / "test_machineguard.db"
os.environ["MACHINEGUARD_DB_PATH"] = str(_DB_PATH)

import database  # noqa: E402

RESULT = {
    "failure_probability": 0.9653,
    "prediction_code": 1,
    "risk_level": "CRITICAL",
    "decision_threshold": 0.35,
    "recommended_action": "Immediate intervention recommended.",
    "model_provenance": {"model_name": "Optimized Random Forest"},
}


def test_create_and_authenticate() -> None:
    database.init_db()

    user = database.create_user("Jane Doe", "Jane@Example.com", "supersecret")
    assert user["id"] > 0
    assert user["name"] == "Jane Doe"
    assert user["email"] == "jane@example.com"

    assert database.authenticate_user("jane@example.com", "supersecret") is not None
    assert database.authenticate_user("JANE@example.com", "supersecret") is not None
    assert database.authenticate_user("jane@example.com", "wrong") is None
    assert database.authenticate_user("missing@example.com", "supersecret") is None


def test_password_is_hashed() -> None:
    conn = sqlite3.connect(_DB_PATH)
    try:
        stored = conn.execute(
            "SELECT password_hash FROM users WHERE email = ?", ("jane@example.com",)
        ).fetchone()[0]
    finally:
        conn.close()

    assert stored != "supersecret"
    assert stored.startswith("pbkdf2_sha256$")
    assert database.verify_password("supersecret", stored)
    assert not database.verify_password("not-it", stored)


def test_duplicate_and_validation() -> None:
    for name, email, password in [
        ("Jane Doe", "jane@example.com", "supersecret"),  # duplicate
        ("X", "x@example.com", "supersecret"),            # short name
        ("Valid Name", "not-an-email", "supersecret"),     # bad email
        ("Valid Name", "valid@example.com", "short"),      # short password
    ]:
        try:
            database.create_user(name, email, password)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected ValueError for {name!r}, {email!r}")


def test_prediction_history() -> None:
    user = database.authenticate_user("jane@example.com", "supersecret")

    first = database.save_prediction(
        user_id=user["id"],
        scenario="High Torque and Tool Wear",
        air_temperature=301.5,
        process_temperature=310.8,
        rotational_speed=1300,
        torque=65.5,
        tool_wear=215,
        result=RESULT,
    )
    second = database.save_prediction(
        user_id=user["id"],
        scenario="Healthy Machine",
        air_temperature=298.2,
        process_temperature=308.7,
        rotational_speed=1500,
        torque=40.0,
        tool_wear=35,
        result={
            **RESULT,
            "failure_probability": 0.02,
            "prediction_code": 0,
            "risk_level": "LOW",
        },
    )

    assert second == first + 1

    rows = database.get_predictions(user["id"])
    assert len(rows) == 2
    assert rows[0]["id"] == second  # newest first
    assert rows[1]["failure_percentage"] == 96.53

    stats = database.get_prediction_stats(user["id"])
    assert stats["total"] == 2
    assert stats["failures"] == 1
    assert stats["normal"] == 1
    assert stats["max_percentage"] == 96.53

    # A different user's history is isolated.
    other = database.create_user("Other User", "other@example.com", "anotherpass")
    assert database.get_predictions(other["id"]) == []

    deleted = database.clear_predictions(user["id"])
    assert deleted == 2
    assert database.get_predictions(user["id"]) == []


def test_bulk_prediction_history() -> None:
    database.init_db()
    user = database.create_user("Batch User", "batch@example.com", "batchpass123")

    records = [
        {
            "scenario": f"Batch upload · MACHINE-{index}",
            "air_temperature": 298.0 + index,
            "process_temperature": 309.0 + index,
            "rotational_speed": 1500,
            "torque": 40.0,
            "tool_wear": 10 + index,
            "failure_probability": 0.9 - index * 0.1,
            "prediction_code": 1,
            "risk_level": "CRITICAL",
            "decision_threshold": 0.35,
            "recommended_action": "Inspect.",
            "model_name": "Optimized Random Forest",
        }
        for index in range(3)
    ]

    assert database.save_predictions_bulk(user["id"], []) == 0
    assert database.save_predictions_bulk(user["id"], records) == 3

    rows = database.get_predictions(user["id"])
    assert len(rows) == 3
    assert [row["scenario"] for row in rows][0] == "Batch upload · MACHINE-2"
    assert all(row["failure_percentage"] > 0 for row in rows)


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
