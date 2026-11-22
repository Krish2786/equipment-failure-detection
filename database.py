"""
Database storage for MachineGuard users and prediction history.
==============================================================

Supports dual backends:
1. Supabase PostgreSQL when DATABASE_URL is configured (via environment variable
   or Streamlit secrets).
2. Local SQLite fallback (data/machineguard.db) for offline development and testing.

Passwords are stored as salted PBKDF2-HMAC-SHA256 hashes, never as plain text.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    from psycopg2 import IntegrityError as PGIntegrityError
    from psycopg2.errors import DeadlockDetected as PGDeadlockDetected
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False
    RealDictCursor = None
    class PGIntegrityError(Exception):
        pass
    class PGDeadlockDetected(Exception):
        pass

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
    machine_id           TEXT    DEFAULT 'UNASSIGNED',
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

PG_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            SERIAL PRIMARY KEY,
    name          TEXT    NOT NULL,
    email         TEXT    NOT NULL UNIQUE,
    password_hash TEXT    NOT NULL,
    created_at    TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS predictions (
    id                   SERIAL PRIMARY KEY,
    user_id              INTEGER NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    machine_id           TEXT    DEFAULT 'UNASSIGNED',
    created_at           TEXT    NOT NULL,
    scenario             TEXT,
    air_temperature      DOUBLE PRECISION NOT NULL,
    process_temperature  DOUBLE PRECISION NOT NULL,
    rotational_speed     DOUBLE PRECISION NOT NULL,
    torque               DOUBLE PRECISION NOT NULL,
    tool_wear            DOUBLE PRECISION NOT NULL,
    failure_probability  DOUBLE PRECISION NOT NULL,
    failure_percentage   DOUBLE PRECISION NOT NULL,
    prediction_code      INTEGER NOT NULL,
    risk_level           TEXT    NOT NULL,
    decision_threshold   DOUBLE PRECISION NOT NULL,
    recommended_action   TEXT,
    model_name           TEXT
);

CREATE INDEX IF NOT EXISTS idx_predictions_user_created
    ON predictions (user_id, created_at DESC);
"""


def get_database_url() -> str | None:
    """Return configured database connection string from environment or Streamlit secrets."""
    url = os.getenv("DATABASE_URL")
    if url and url.strip():
        return url.strip()
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "DATABASE_URL" in st.secrets:
            val = st.secrets["DATABASE_URL"]
            if val and str(val).strip():
                return str(val).strip()
    except Exception:
        pass
    return None


class DatabaseSession:
    """Unified wrapper around SQLite and PostgreSQL connections and queries."""

    def __init__(self, is_pg: bool, conn: Any):
        self.is_pg = is_pg
        self.conn = conn

    def execute(self, sql: str, params: tuple | list = ()):
        if self.is_pg:
            pg_sql = sql.replace("?", "%s")
            cur = self.conn.cursor()
            cur.execute(pg_sql, params)
            return cur
        else:
            return self.conn.execute(sql, params)

    def executemany(self, sql: str, params_seq):
        if self.is_pg:
            pg_sql = sql.replace("?", "%s")
            cur = self.conn.cursor()
            cur.executemany(pg_sql, params_seq)
            return cur
        else:
            return self.conn.executemany(sql, params_seq)

    def executescript(self, script: str):
        if self.is_pg:
            cur = self.conn.cursor()
            cur.execute(script)
            return cur
        else:
            return self.conn.executescript(script)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.conn, name)


@contextmanager
def _connection() -> Iterator[DatabaseSession]:
    db_url = get_database_url()
    if db_url:
        if not HAS_PSYCOPG2:
            raise RuntimeError(
                "DATABASE_URL is set but psycopg2 is not installed. "
                "Please run `pip install psycopg2-binary`."
            )
        conn = psycopg2.connect(db_url, sslmode="require", cursor_factory=RealDictCursor)
        try:
            yield DatabaseSession(is_pg=True, conn=conn)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    else:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield DatabaseSession(is_pg=False, conn=conn)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()


def init_db() -> None:
    """Create the schema if it does not exist. Safe to call on every run."""
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            with _connection() as conn:
                if conn.is_pg:
                    cur = conn.conn.cursor()
                    # 1. Acquire transaction-scoped advisory lock to serialize concurrent worker DDL execution
                    cur.execute("SELECT pg_advisory_xact_lock(78234106);")
                    cur.execute(PG_SCHEMA)

                    # 2. Check information_schema before attempting ALTER TABLE
                    cur.execute(
                        """
                        SELECT 1 FROM information_schema.columns
                        WHERE table_schema = 'public'
                          AND table_name = 'predictions'
                          AND column_name = 'machine_id'
                        """
                    )
                    has_machine_id = cur.fetchone() is not None
                    if not has_machine_id:
                        cur.execute(
                            "ALTER TABLE predictions ADD COLUMN IF NOT EXISTS machine_id TEXT DEFAULT 'UNASSIGNED'"
                        )
                    cur.execute(
                        "CREATE INDEX IF NOT EXISTS idx_predictions_user_machine "
                        "ON predictions (user_id, machine_id, created_at)"
                    )
                else:
                    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
                    conn.conn.executescript(SCHEMA)
                    # Backward-compatible check for legacy databases without machine_id
                    cursor = conn.conn.execute("PRAGMA table_info(predictions)")
                    columns = [row["name"] for row in cursor.fetchall()]
                    if "machine_id" not in columns:
                        conn.conn.execute(
                            "ALTER TABLE predictions ADD COLUMN machine_id TEXT DEFAULT 'UNASSIGNED'"
                        )
                    conn.conn.execute(
                        "CREATE INDEX IF NOT EXISTS idx_predictions_user_machine "
                        "ON predictions (user_id, machine_id, created_at)"
                    )
            return
        except PGDeadlockDetected:
            if attempt < max_attempts:
                time.sleep(0.2)
            else:
                raise


def normalize_machine_id(machine_id: Any) -> str:
    """Normalize asset identifiers to uppercase string, defaulting to 'UNASSIGNED'."""
    if machine_id is None:
        return "UNASSIGNED"
    val = str(machine_id).strip().upper()
    if not val or val in {"NAN", "NONE", "<NA>", "NULL"}:
        return "UNASSIGNED"
    return val


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


def _row_to_user(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "name": row["name"],
        "email": row["email"],
        "created_at": str(row["created_at"]),
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
                RETURNING id
                """,
                (name, email, hash_password(password), created_at),
            )
            inserted = cursor.fetchone()
            user_id = int(inserted["id"] if (isinstance(inserted, dict) or hasattr(inserted, "keys")) else inserted[0])
            row = conn.execute(
                "SELECT id, name, email, created_at FROM users WHERE id = ?",
                (user_id,),
            ).fetchone()
    except (sqlite3.IntegrityError, PGIntegrityError) as exc:
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
    machine_id: str = "UNASSIGNED",
) -> int:
    """Persist one model prediction for a user and return the row id."""
    created_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
    norm_machine_id = normalize_machine_id(machine_id)
    with _connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO predictions (
                user_id, machine_id, created_at, scenario,
                air_temperature, process_temperature, rotational_speed, torque, tool_wear,
                failure_probability, failure_percentage, prediction_code, risk_level,
                decision_threshold, recommended_action, model_name
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            RETURNING id
            """,
            (
                user_id,
                norm_machine_id,
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
        inserted = cursor.fetchone()
        return int(inserted["id"] if (isinstance(inserted, dict) or hasattr(inserted, "keys")) else inserted[0])


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
            normalize_machine_id(record.get("machine_id")),
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
                user_id, machine_id, created_at, scenario,
                air_temperature, process_temperature, rotational_speed, torque, tool_wear,
                failure_probability, failure_percentage, prediction_code, risk_level,
                decision_threshold, recommended_action, model_name
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
    return len(rows)


def get_predictions(
    user_id: int,
    limit: int | None = None,
    machine_id: str | None = None,
) -> list[dict[str, Any]]:
    query = """
        SELECT * FROM predictions
        WHERE user_id = ?
    """
    params: list[Any] = [user_id]
    if machine_id is not None and machine_id != "All":
        query += " AND machine_id = ?"
        params.append(normalize_machine_id(machine_id))

    query += " ORDER BY id DESC"
    if limit is not None:
        query += " LIMIT ?"
        params.append(int(limit))

    with _connection() as conn:
        rows = conn.execute(query, params).fetchall()
    return [
        {**dict(row), "machine_id": row["machine_id"] or "UNASSIGNED"}
        for row in rows
    ]


def get_prediction(user_id: int, prediction_id: int) -> dict[str, Any] | None:
    with _connection() as conn:
        row = conn.execute(
            "SELECT * FROM predictions WHERE id = ? AND user_id = ?",
            (prediction_id, user_id),
        ).fetchone()
    if not row:
        return None
    res = dict(row)
    res["machine_id"] = res.get("machine_id") or "UNASSIGNED"
    return res


def get_machine_trend(user_id: int, machine_id: str) -> list[dict[str, Any]]:
    """Return all predictions for a specific machine ordered chronologically (created_at ASC)."""
    norm_id = normalize_machine_id(machine_id)
    query = """
        SELECT
            id,
            user_id,
            machine_id,
            created_at,
            scenario,
            air_temperature,
            process_temperature,
            rotational_speed,
            torque,
            tool_wear,
            failure_probability,
            failure_percentage,
            prediction_code,
            risk_level,
            decision_threshold,
            recommended_action,
            model_name
        FROM predictions
        WHERE user_id = ? AND machine_id = ?
        ORDER BY created_at ASC, id ASC
    """
    with _connection() as conn:
        rows = conn.execute(query, (user_id, norm_id)).fetchall()
    return [
        {**dict(row), "machine_id": row["machine_id"] or "UNASSIGNED"}
        for row in rows
    ]


def get_distinct_machines(user_id: int) -> list[str]:
    """Return sorted list of distinct machine IDs saved for this user."""
    query = """
        SELECT DISTINCT machine_id
        FROM predictions
        WHERE user_id = ? AND machine_id IS NOT NULL AND machine_id != ''
        ORDER BY machine_id ASC
    """
    with _connection() as conn:
        rows = conn.execute(query, (user_id,)).fetchall()
    machines = [row["machine_id"] for row in rows if row["machine_id"]]
    return machines if machines else ["UNASSIGNED"]


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


def get_fleet_summary(user_id: int) -> list[dict[str, Any]]:
    """Return one summary row per distinct machine for the fleet overview.

    Each row contains:
        machine_id, total_checks, failure_checks, last_check_at,
        latest_risk, latest_failure_pct, avg_failure_pct, peak_failure_pct,
        first_failure_pct, latest_tool_wear, first_tool_wear,
        latest_rotational_speed, latest_torque
    """
    query = """
        SELECT
            machine_id,
            COUNT(*)                              AS total_checks,
            COALESCE(SUM(prediction_code), 0)     AS failure_checks,
            MAX(created_at)                       AS last_check_at,
            COALESCE(AVG(failure_percentage), 0)  AS avg_failure_pct,
            COALESCE(MAX(failure_percentage), 0)  AS peak_failure_pct
        FROM predictions
        WHERE user_id = ?
          AND machine_id IS NOT NULL
          AND machine_id != ''
        GROUP BY machine_id
        ORDER BY machine_id ASC
    """
    with _connection() as conn:
        rows = conn.execute(query, (user_id,)).fetchall()
        summaries = []
        for row in rows:
            mid = row["machine_id"] or "UNASSIGNED"
            # Latest record for this machine
            latest = conn.execute(
                """
                SELECT failure_percentage, risk_level, tool_wear,
                       rotational_speed, torque, created_at
                FROM predictions
                WHERE user_id = ? AND machine_id = ?
                ORDER BY created_at DESC, id DESC
                LIMIT 1
                """,
                (user_id, mid),
            ).fetchone()
            # Earliest record for trend direction
            first = conn.execute(
                """
                SELECT failure_percentage, tool_wear
                FROM predictions
                WHERE user_id = ? AND machine_id = ?
                ORDER BY created_at ASC, id ASC
                LIMIT 1
                """,
                (user_id, mid),
            ).fetchone()
            summaries.append({
                "machine_id": mid,
                "total_checks": int(row["total_checks"] or 0),
                "failure_checks": int(row["failure_checks"] or 0),
                "last_check_at": row["last_check_at"] or "",
                "avg_failure_pct": round(float(row["avg_failure_pct"] or 0), 2),
                "peak_failure_pct": round(float(row["peak_failure_pct"] or 0), 2),
                "latest_failure_pct": round(float(latest["failure_percentage"] if latest else 0), 2),
                "latest_risk": latest["risk_level"] if latest else "UNKNOWN",
                "latest_tool_wear": float(latest["tool_wear"] if latest else 0),
                "latest_rotational_speed": float(latest["rotational_speed"] if latest else 0),
                "latest_torque": float(latest["torque"] if latest else 0),
                "first_failure_pct": round(float(first["failure_percentage"] if first else 0), 2),
                "first_tool_wear": float(first["tool_wear"] if first else 0),
            })
    return summaries


def reassign_machine_id(
    user_id: int,
    old_machine_id: str,
    new_machine_id: str,
    prediction_ids: list[int] | None = None,
) -> int:
    """Update machine_id for records belonging to user_id.

    If prediction_ids is provided, update only those specific record IDs.
    Otherwise, update all records matching old_machine_id for user_id.
    Returns the count of updated records.
    """
    norm_new = normalize_machine_id(new_machine_id)
    norm_old = normalize_machine_id(old_machine_id)

    with _connection() as conn:
        if prediction_ids is not None:
            if not prediction_ids:
                return 0
            placeholders = ",".join("?" for _ in prediction_ids)
            cursor = conn.execute(
                f"""
                UPDATE predictions
                SET machine_id = ?
                WHERE user_id = ? AND id IN ({placeholders})
                """,
                [norm_new, user_id, *[int(i) for i in prediction_ids]],
            )
        else:
            cursor = conn.execute(
                """
                UPDATE predictions
                SET machine_id = ?
                WHERE user_id = ? AND machine_id = ?
                """,
                (norm_new, user_id, norm_old),
            )
        return cursor.rowcount


def clear_predictions(user_id: int) -> int:
    with _connection() as conn:
        cursor = conn.execute(
            "DELETE FROM predictions WHERE user_id = ?", (user_id,)
        )
        return cursor.rowcount
