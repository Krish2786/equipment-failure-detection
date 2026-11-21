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
            "machine_id": f"MILL-{index + 1}",
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
    assert {row["machine_id"] for row in rows} == {"MILL-1", "MILL-2", "MILL-3"}


def test_machine_id_tracking_and_filtering() -> None:
    database.init_db()
    user = database.create_user("Asset User", "asset@example.com", "assetpassword")

    # 1. Normalized uppercase machine_id
    id1 = database.save_prediction(
        user_id=user["id"],
        scenario="Mill inspection 1",
        air_temperature=300.0,
        process_temperature=310.0,
        rotational_speed=1400,
        torque=50.0,
        tool_wear=40.0,
        result=RESULT,
        machine_id="mill-01",
    )
    p1 = database.get_prediction(user["id"], id1)
    assert p1["machine_id"] == "MILL-01"

    # 2. Default to UNASSIGNED when empty or None
    id2 = database.save_prediction(
        user_id=user["id"],
        scenario="Unassigned inspection",
        air_temperature=300.0,
        process_temperature=310.0,
        rotational_speed=1400,
        torque=50.0,
        tool_wear=40.0,
        result=RESULT,
        machine_id="   ",
    )
    p2 = database.get_prediction(user["id"], id2)
    assert p2["machine_id"] == "UNASSIGNED"

    # 3. Filtering by machine_id
    mill_rows = database.get_predictions(user["id"], machine_id="MILL-01")
    assert len(mill_rows) == 1
    assert mill_rows[0]["machine_id"] == "MILL-01"

    all_rows = database.get_predictions(user["id"], machine_id="All")
    assert len(all_rows) == 2

    # 4. Distinct machines list
    machines = database.get_distinct_machines(user["id"])
    assert "MILL-01" in machines
    assert "UNASSIGNED" in machines


def test_machine_trend_chronological() -> None:
    database.init_db()
    user = database.create_user("Trend User", "trend@example.com", "trendpassword")

    # Insert sequential checks with increasing wear and risk
    for i, (wear, prob) in enumerate([(20.0, 0.05), (80.0, 0.25), (210.0, 0.88)]):
        database.save_prediction(
            user_id=user["id"],
            scenario=f"Check {i + 1}",
            air_temperature=298.0 + i,
            process_temperature=308.0 + i,
            rotational_speed=1500 - i * 50,
            torque=40.0 + i * 10,
            tool_wear=wear,
            result={
                **RESULT,
                "failure_probability": prob,
                "risk_level": "LOW" if prob < 0.2 else ("MODERATE" if prob < 0.35 else "CRITICAL"),
            },
            machine_id="SPINDLE-4",
        )

    # Insert a different machine check
    database.save_prediction(
        user_id=user["id"],
        scenario="Other machine",
        air_temperature=300.0,
        process_temperature=310.0,
        rotational_speed=1500,
        torque=40.0,
        tool_wear=10.0,
        result=RESULT,
        machine_id="PRESS-99",
    )

    trend = database.get_machine_trend(user["id"], "SPINDLE-4")
    assert len(trend) == 3
    assert all(r["machine_id"] == "SPINDLE-4" for r in trend)

    # Chronologically ordered
    assert trend[0]["id"] < trend[1]["id"] < trend[2]["id"]
    assert trend[0]["tool_wear"] == 20.0
    assert trend[1]["tool_wear"] == 80.0
    assert trend[2]["tool_wear"] == 210.0

    # Required fields present
    first = trend[0]
    for key in ["id", "created_at", "failure_percentage", "tool_wear", "torque", "rotational_speed", "risk_level"]:
        assert key in first, f"Missing key {key} in trend item"


def test_fleet_summary() -> None:
    database.init_db()
    user = database.create_user("Fleet User", "fleet@example.com", "fleetpassword")

    # Save multiple predictions across 2 machines
    # Machine 1: M-01 (2 runs: low then critical)
    database.save_prediction(
        user_id=user["id"],
        scenario="Run 1",
        air_temperature=298.0,
        process_temperature=308.0,
        rotational_speed=1500,
        torque=40.0,
        tool_wear=20.0,
        result={**RESULT, "failure_probability": 0.10, "prediction_code": 0, "risk_level": "LOW"},
        machine_id="M-01",
    )
    database.save_prediction(
        user_id=user["id"],
        scenario="Run 2",
        air_temperature=303.0,
        process_temperature=312.0,
        rotational_speed=1300,
        torque=65.0,
        tool_wear=220.0,
        result={**RESULT, "failure_probability": 0.90, "prediction_code": 1, "risk_level": "CRITICAL"},
        machine_id="M-01",
    )

    # Machine 2: M-02 (1 run: moderate)
    database.save_prediction(
        user_id=user["id"],
        scenario="Run 3",
        air_temperature=300.0,
        process_temperature=310.0,
        rotational_speed=1450,
        torque=48.0,
        tool_wear=75.0,
        result={**RESULT, "failure_probability": 0.28, "prediction_code": 0, "risk_level": "MODERATE"},
        machine_id="M-02",
    )

    fleet = database.get_fleet_summary(user["id"])
    assert len(fleet) == 2
    fleet_dict = {m["machine_id"]: m for m in fleet}

    assert "M-01" in fleet_dict
    assert "M-02" in fleet_dict

    m1 = fleet_dict["M-01"]
    assert m1["total_checks"] == 2
    assert m1["failure_checks"] == 1  # 2nd was CRITICAL with prediction_code 1
    assert m1["latest_risk"] == "CRITICAL"
    assert m1["latest_failure_pct"] == 90.0
    assert m1["first_failure_pct"] == 10.0
    assert m1["latest_tool_wear"] == 220.0
    assert m1["peak_failure_pct"] == 90.0
    assert m1["avg_failure_pct"] == 50.0

    m2 = fleet_dict["M-02"]
    assert m2["total_checks"] == 1
    assert m2["latest_risk"] == "MODERATE"
    assert m2["latest_failure_pct"] == 28.0
    assert m2["latest_tool_wear"] == 75.0


def test_legacy_database_migration() -> None:
    """Ensure databases created prior to machine_id are upgraded without dropping records."""
    temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    legacy_db_path = Path(temp_dir.name) / "legacy.db"
    orig_path = database.DB_PATH
    database.DB_PATH = legacy_db_path

    try:
        # Create legacy table without machine_id column
        conn = sqlite3.connect(legacy_db_path)
        conn.executescript("""
        CREATE TABLE users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            scenario TEXT,
            air_temperature REAL NOT NULL,
            process_temperature REAL NOT NULL,
            rotational_speed REAL NOT NULL,
            torque REAL NOT NULL,
            tool_wear REAL NOT NULL,
            failure_probability REAL NOT NULL,
            failure_percentage REAL NOT NULL,
            prediction_code INTEGER NOT NULL,
            risk_level TEXT NOT NULL,
            decision_threshold REAL NOT NULL,
            recommended_action TEXT,
            model_name TEXT
        );
        """)
        conn.execute(
            "INSERT INTO users (name, email, password_hash, created_at) VALUES ('Old User', 'old@ex.com', 'hash', '2025-01-01')"
        )
        conn.execute(
            """INSERT INTO predictions (
                user_id, created_at, scenario, air_temperature, process_temperature,
                rotational_speed, torque, tool_wear, failure_probability, failure_percentage,
                prediction_code, risk_level, decision_threshold
            ) VALUES (1, '2025-01-01 10:00:00', 'Old Run', 298.0, 308.0, 1500, 40, 30, 0.05, 5.0, 0, 'LOW', 0.35)"""
        )
        conn.commit()
        conn.close()

        # Call init_db() on the legacy database
        database.init_db()

        # Verify column added
        conn = sqlite3.connect(legacy_db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.execute("PRAGMA table_info(predictions)")
        columns = [row["name"] for row in cursor.fetchall()]
        assert "machine_id" in columns

        # Verify old record preserved with default 'UNASSIGNED'
        legacy_row = conn.execute("SELECT * FROM predictions WHERE id = 1").fetchone()
        assert legacy_row["machine_id"] == "UNASSIGNED"
        assert legacy_row["scenario"] == "Old Run"
        conn.close()

        # Verify saving new prediction works on migrated database
        new_id = database.save_prediction(
            user_id=1,
            scenario="New Run",
            air_temperature=299.0,
            process_temperature=309.0,
            rotational_speed=1500,
            torque=40.0,
            tool_wear=50.0,
            result=RESULT,
            machine_id="MIGRATED-01",
        )
        p = database.get_prediction(1, new_id)
        assert p["machine_id"] == "MIGRATED-01"

    finally:
        database.DB_PATH = orig_path
        temp_dir.cleanup()


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
