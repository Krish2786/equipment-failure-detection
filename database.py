"""
SQLite storage for MachineGuard users and prediction history.
=============================================================

Passwords are stored as salted PBKDF2-HMAC-SHA256 hashes, never as plain text.
The database lives at ``data/machineguard.db`` by default. Set the
``MACHINEGUARD_DB_PATH`` environment variable to use a different file (the tests
do this to keep the real database untouched).

This is intentionally small and dependency-free: SQLite, hashlib, secrets and
hmac all ship with Python.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

PROJECT_ROOT = Path(__file__).resolve().parent
DB_PATH = Path(os.getenv("MACHINEGUARD_DB_PATH", PROJECT_ROOT / "data" / "machineguard.db"))

PBKDF2_ALGORITHM = "pbkdf2_sha256"
PBKDF2_ITERATIONS = 260_000
MIN_PASSWORD_LENGTH = 8

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT    NOT NULL,
    email         TEXT    NOT NULL UNIQUE COLLATE NOCASE,
    password_hash TEXT    NOT NULL,
    created_at    TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS predictions (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id              INTEGER NOT NULL,
    created_at           TEXT    NOT NULL,
    scenario             TEXT,
    air_temperature      REAL    NOT NULL,
    process_temperature  REAL    NOT NULL,
    rotational_speed     REAL    NOT NULL,
    torque               REAL    NOT NULL,
    tool_wear            REAL    NOT NULL,
    failure_probability  REAL    NOT NULL,
    failure_percentage   REAL    NOT NULL,
    prediction_code      INTEGER NOT NULL,
    risk_level           TEXT    NOT NULL,
    decision_threshold   REAL    NOT NULL,
    recommended_action   TEXT,
    model_name           TEXT,
    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_predictions_user_created
    ON predictions (user_id, created_at DESC);
"""


@contextmanager
def _connection() -> Iterator[sqlite3.Connection]:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """Create the schema if it does not exist. Safe to call on every run."""
    with _connection() as conn:
        conn.executescript(SCHEMA)


# ---------------------------------------------------------------------------
# Passwords
# ---------------------------------------------------------------------------


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS
    )
    return f"{PBKDF2_ALGORITHM}${PBKDF2_ITERATIONS}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algorithm, iterations, salt_hex, digest_hex = stored.split("$")
        if algorithm != PBKDF2_ALGORITHM:
            return False
        digest = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            bytes.fromhex(salt_hex),
            int(iterations),
        )
        return hmac.compare_digest(digest.hex(), digest_hex)
    except (ValueError, TypeError):
        return False


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------


def _row_to_user(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "name": row["name"],
        "email": row["email"],
        "created_at": row["created_at"],
    }


def create_user(name: str, email: str, password: str) -> dict[str, Any]:
    """Create a user and return the public record. Raises ValueError on bad input."""
    name = (name or "").strip()
    email = (email or "").strip().lower()

    if len(name) < 2:
        raise ValueError("Please enter your full name.")
    if "@" not in email or "." not in email.split("@")[-1]:
        raise ValueError("Please enter a valid email address.")
    if len(password or "") < MIN_PASSWORD_LENGTH:
        raise ValueError(
            f"Password must be at least {MIN_PASSWORD_LENGTH} characters long."
        )

    created_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with _connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO users (name, email, password_hash, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (name, email, hash_password(password), created_at),
            )
            user_id = int(cursor.lastrowid)
            row = conn.execute(
                "SELECT id, name, email, created_at FROM users WHERE id = ?",
                (user_id,),
            ).fetchone()
    except sqlite3.IntegrityError as exc:
        raise ValueError("An account with this email already exists.") from exc

    return _row_to_user(row)


def authenticate_user(email: str, password: str) -> dict[str, Any] | None:
    """Return the user when email + password match, otherwise None."""
    email = (email or "").strip().lower()
    with _connection() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE email = ?", (email,)
        ).fetchone()

    if row is None or not verify_password(password or "", row["password_hash"]):
        return None
    return _row_to_user(row)


def get_user_by_id(user_id: int) -> dict[str, Any] | None:
    with _connection() as conn:
        row = conn.execute(
            "SELECT id, name, email, created_at FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()
    return _row_to_user(row) if row else None


# ---------------------------------------------------------------------------
# Prediction history
# ---------------------------------------------------------------------------


def save_prediction(
    user_id: int,
    scenario: str,
    air_temperature: float,
    process_temperature: float,
    rotational_speed: float,
    torque: float,
    tool_wear: float,
    result: dict[str, Any],
) -> int:
    """Persist one model prediction for a user and return the row id."""
    created_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
    with _connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO predictions (
                user_id, created_at, scenario,
                air_temperature, process_temperature, rotational_speed, torque, tool_wear,
                failure_probability, failure_percentage, prediction_code, risk_level,
                decision_threshold, recommended_action, model_name
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                created_at,
                scenario,
                float(air_temperature),
                float(process_temperature),
                float(rotational_speed),
                float(torque),
                float(tool_wear),
                float(result["failure_probability"]),
                round(float(result["failure_probability"]) * 100, 2),
                int(result["prediction_code"]),
                str(result["risk_level"]),
                float(result["decision_threshold"]),
                result.get("recommended_action"),
                (result.get("model_provenance") or {}).get("model_name"),
            ),
        )
        return int(cursor.lastrowid)


def save_predictions_bulk(
    user_id: int,
    records: list[dict[str, Any]],
) -> int:
    """Persist many predictions for a user in one transaction.

    Each record is a flat mapping with the sensor readings plus the model
    output, e.g. the dictionaries produced for a batch upload. Invalid rows
    (no probability) must be filtered out by the caller. Returns the number of
    rows inserted.
    """
    if not records:
        return 0

    created_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
    rows = [
        (
            user_id,
            created_at,
            record.get("scenario") or "Batch Upload",
            float(record["air_temperature"]),
            float(record["process_temperature"]),
            float(record["rotational_speed"]),
            float(record["torque"]),
            float(record["tool_wear"]),
            float(record["failure_probability"]),
            round(float(record["failure_probability"]) * 100, 2),
            int(record["prediction_code"]),
            str(record["risk_level"]),
            float(record["decision_threshold"]),
            record.get("recommended_action"),
            record.get("model_name"),
        )
        for record in records
    ]

    with _connection() as conn:
        conn.executemany(
            """
            INSERT INTO predictions (
                user_id, created_at, scenario,
                air_temperature, process_temperature, rotational_speed, torque, tool_wear,
                failure_probability, failure_percentage, prediction_code, risk_level,
                decision_threshold, recommended_action, model_name
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
    return len(rows)


def get_predictions(user_id: int, limit: int | None = None) -> list[dict[str, Any]]:
    query = """
        SELECT * FROM predictions
        WHERE user_id = ?
        ORDER BY id DESC
    """
    params: list[Any] = [user_id]
    if limit is not None:
        query += " LIMIT ?"
        params.append(int(limit))

    with _connection() as conn:
        rows = conn.execute(query, params).fetchall()
    return [dict(row) for row in rows]


def get_prediction(user_id: int, prediction_id: int) -> dict[str, Any] | None:
    with _connection() as conn:
        row = conn.execute(
            "SELECT * FROM predictions WHERE id = ? AND user_id = ?",
            (prediction_id, user_id),
        ).fetchone()
    return dict(row) if row else None


def get_prediction_stats(user_id: int) -> dict[str, Any]:
    with _connection() as conn:
        row = conn.execute(
            """
            SELECT
                COUNT(*)                                   AS total,
                COALESCE(SUM(prediction_code), 0)          AS failures,
                COALESCE(AVG(failure_percentage), 0)       AS avg_percentage,
                COALESCE(MAX(failure_percentage), 0)       AS max_percentage,
                MAX(created_at)                            AS last_prediction_at
            FROM predictions
            WHERE user_id = ?
            """,
            (user_id,),
        ).fetchone()

    total = int(row["total"] or 0)
    return {
        "total": total,
        "failures": int(row["failures"] or 0),
        "normal": total - int(row["failures"] or 0),
        "avg_percentage": round(float(row["avg_percentage"] or 0), 2),
        "max_percentage": round(float(row["max_percentage"] or 0), 2),
        "last_prediction_at": row["last_prediction_at"],
    }


def clear_predictions(user_id: int) -> int:
    with _connection() as conn:
        cursor = conn.execute(
            "DELETE FROM predictions WHERE user_id = ?", (user_id,)
        )
        return cursor.rowcount
