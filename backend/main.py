"""
MachineGuard - FastAPI inference service
========================================

Serves the trained AI4I 2020 Random Forest model
(`ml/models/best_model/ai4i_failure_model.joblib`) over HTTP so the React
frontend can score real telemetry instead of using the JS heuristic emulator.

Endpoints
---------
GET  /                          Service index
GET  /health                    Liveness + model status
GET  /api/model/metadata        Model card, metrics, feature importance
POST /api/predict               Score one sensor reading
POST /api/predict/batch         Score many sensor readings
GET  /docs                      Swagger UI (automatic)
GET  /openapi.json              OpenAPI schema (automatic)

Run from the project root:
    python -m uvicorn backend.main:app --reload --port 8000

Run from this directory:
    python -m uvicorn main:app --reload --port 8000
"""

from __future__ import annotations

import logging
import math
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.alias_generators import to_camel

# ---------------------------------------------------------------------------
# Paths & model loading
# ---------------------------------------------------------------------------

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
ML_SRC_DIR = PROJECT_ROOT / "ml" / "src"
MODEL_DIR = PROJECT_ROOT / "ml" / "models" / "best_model"

# The serialized pipeline references the `utils` module, so ml/src must be
# importable before joblib unpickles the model.
if str(ML_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(ML_SRC_DIR))

from model_prediction import EquipmentFailurePredictor  # noqa: E402

logging.basicConfig(
    level=os.getenv("MACHINEGUARD_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("machineguard.api")

MODEL_VERSION = os.getenv("MACHINEGUARD_MODEL_VERSION", "rf-ai4i-optimized-v1")
SERVICE_STARTED_AT = time.time()


@lru_cache(maxsize=1)
def get_predictor() -> EquipmentFailurePredictor:
    """Load the production model once and reuse it across requests."""
    logger.info("Loading model from %s", MODEL_DIR)
    predictor = EquipmentFailurePredictor(model_dir=MODEL_DIR)
    logger.info(
        "Model ready: %s (threshold=%.2f)",
        predictor.model_name,
        predictor.threshold,
    )
    return predictor


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class CamelModel(BaseModel):
    """Base model: accepts snake_case, serializes as camelCase for the UI."""

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
    )


class SensorReading(CamelModel):
    """The five physical telemetry inputs understood by the model."""

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        extra="forbid",
    )

    air_temperature: float = Field(
        ...,
        ge=290.0,
        le=310.0,
        description="Ambient air temperature [K]",
        examples=[298.4],
    )
    process_temperature: float = Field(
        ...,
        ge=300.0,
        le=320.0,
        description="Process/cutting-zone temperature [K]",
        examples=[308.7],
    )
    rotational_speed: float = Field(
        ...,
        ge=1000.0,
        le=3000.0,
        description="Spindle rotational speed [rpm]",
        examples=[1500],
    )
    torque: float = Field(
        ...,
        ge=3.0,
        le=80.0,
        description="Spindle torque [Nm]",
        examples=[40.2],
    )
    tool_wear: float = Field(
        ...,
        ge=0.0,
        le=260.0,
        description="Cumulative tool wear [min]",
        examples=[120],
    )

    @model_validator(mode="after")
    def _check_thermal_gradient(self) -> "SensorReading":
        if self.process_temperature < self.air_temperature:
            raise ValueError(
                "processTemperature must be greater than or equal to "
                "airTemperature during active operation"
            )
        return self


class PredictRequest(SensorReading):
    """One reading plus an optional custom decision threshold."""

    threshold: float | None = Field(
        default=None,
        ge=0.01,
        le=0.99,
        description="Override the model's calibrated 0.35 decision threshold",
    )


class BatchPredictRequest(CamelModel):
    """Score a list of readings in one call."""

    readings: list[SensorReading] = Field(..., min_length=1, max_length=5000)
    threshold: float | None = Field(default=None, ge=0.01, le=0.99)


class SensorsOut(CamelModel):
    air_temperature: float
    process_temperature: float
    rotational_speed: float
    torque: float
    tool_wear: float
    delta_t: float
    power_watts: float


class PredictionResponse(CamelModel):
    id: str
    sample_id: str
    prediction_code: int
    prediction_label: str
    failure_probability: float
    failure_percentage: float
    decision_threshold: float
    risk_level: str
    risk_tier: str
    recommended_action: str
    contributing_factors: list[str]
    sensors: SensorsOut
    model_version: str
    model_provenance: dict[str, Any]
    explanation_source: str
    is_demo: bool
    source: str
    timestamp: str
    disclaimer: str


class BatchPredictionResponse(CamelModel):
    count: int
    model_version: str
    results: list[PredictionResponse]


# ---------------------------------------------------------------------------
# Risk tiers (kept in sync with src/utils/formatters.js)
# ---------------------------------------------------------------------------

RISK_TIERS: list[tuple[float, str, str, str]] = [
    (
        25.0,
        "Low Risk",
        "optimal",
        "Routine monitoring active. No intervention required.",
    ),
    (
        50.0,
        "Moderate Risk",
        "moderate",
        "Schedule physical inspection during the next planned maintenance window.",
    ),
    (
        75.0,
        "High Risk",
        "high",
        "Prioritize preventive maintenance. Inspect cutting tool, bearings, and thermal loops.",
    ),
    (
        math.inf,
        "Critical Risk",
        "critical",
        "Immediate intervention recommended. Halt cycle or dispatch maintenance engineer immediately.",
    ),
]


def classify_risk(failure_percentage: float) -> tuple[str, str, str]:
    for upper_bound, tier, level, action in RISK_TIERS:
        if failure_percentage < upper_bound:
            return tier, level, action
    # Unreachable because the final bound is infinity.
    return "Critical Risk", "critical", RISK_TIERS[-1][3]


# ---------------------------------------------------------------------------
# Physics-based explanation helpers
# ---------------------------------------------------------------------------
# The Random Forest exposes probabilities, not per-reading explanations. These
# factors restate the known AI4I physics thresholds so the UI has something
# concrete to show next to the model score.

def physical_contributing_factors(reading: SensorReading) -> list[str]:
    air = reading.air_temperature
    process = reading.process_temperature
    speed = reading.rotational_speed
    torque = reading.torque
    tool_wear = reading.tool_wear

    delta_t = process - air
    power_watts = torque * speed * (2.0 * math.pi / 60.0)
    overstrain_index = torque * tool_wear

    factors: list[str] = []

    # Heat dissipation failure (HDF)
    if delta_t < 8.6:
        factors.append(
            f"Heat dissipation deficiency: Delta-T is low ({delta_t:.1f} K < 8.6 K "
            "nominal limit), causing internal heat accumulation"
        )
    elif delta_t > 12.0:
        factors.append(
            f"Elevated process thermal gradient: Delta-T is high ({delta_t:.1f} K > "
            "12.0 K threshold), indicating cooling system strain"
        )

    if process >= 313.0:
        factors.append(
            f"Critical process temperature ({process:.1f} K) near upper structural boundary"
        )

    # Power failure (PWF) and near-stall overload
    if power_watts > 8500:
        factors.append(
            f"Excessive mechanical spindle power ({round(power_watts):,} W > "
            "8,500 W duty rating)"
        )
    elif power_watts < 2200 and torque > 45:
        factors.append(
            f"Spindle near-stall overload detected (high torque {torque:.1f} Nm "
            f"at low speed {round(speed)} rpm)"
        )

    if torque > 62.0:
        factors.append(
            f"Extreme mechanical torque load ({torque:.1f} Nm exceeding high-strain "
            "limit of 62.0 Nm)"
        )

    # Overstrain failure (OSF)
    if overstrain_index > 11000:
        factors.append(
            "Critical torque-tool wear overstrain index "
            f"({round(overstrain_index):,} > 11,000 threshold)"
        )

    # Tool wear failure (TWF)
    if tool_wear >= 210:
        factors.append(
            f"Severe cutting tool wear degradation ({tool_wear:.0f} min > 210 min "
            "replacement threshold)"
        )
    elif tool_wear >= 175:
        factors.append(
            f"Elevated tool wear ({tool_wear:.0f} min); tool nearing end of optimal "
            "blade sharpness window"
        )

    # Rotational speed extremes
    if speed >= 2650:
        factors.append(
            f"High spindle velocity ({round(speed)} rpm) inducing centrifugal "
            "bearing vibrations"
        )
    elif speed <= 1150 and torque >= 55:
        factors.append(
            f"High torque at low rotational velocity ({round(speed)} rpm) risking "
            "motor stator overheating"
        )

    if not factors:
        factors.append(
            "All sensor telemetry parameters operate well within nominal baseline envelopes"
        )
        factors.append(
            "Thermal dissipation rate, torque strain, and tool wear indices are within "
            "safety limits"
        )

    return factors


def feature_importance(predictor: EquipmentFailurePredictor) -> list[dict[str, Any]]:
    """Read Gini importances from the final Random Forest pipeline step."""
    names = predictor.metadata.get("total_internal_features", [])
    model = predictor.model

    importances = None
    if hasattr(model, "named_steps") and model.named_steps:
        final_step = list(model.named_steps.values())[-1]
        importances = getattr(final_step, "feature_importances_", None)
    if importances is None:
        importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return []

    return [
        {
            "feature": name,
            "importance": round(float(value), 6),
            "percentage": round(float(value) * 100.0, 2),
        }
        for name, value in zip(names, importances)
    ]


# ---------------------------------------------------------------------------
# Response assembly
# ---------------------------------------------------------------------------


def build_prediction(
    predictor: EquipmentFailurePredictor,
    reading: SensorReading,
    threshold: float | None = None,
) -> PredictionResponse:
    decision_threshold = (
        float(threshold) if threshold is not None else float(predictor.threshold)
    )

    raw = predictor.predict(
        air_temperature=reading.air_temperature,
        process_temperature=reading.process_temperature,
        rotational_speed=reading.rotational_speed,
        torque=reading.torque,
        tool_wear=reading.tool_wear,
    )

    probability = float(raw["failure_probability"])
    percentage = round(probability * 100.0, 1)
    tier, level, action = classify_risk(percentage)
    is_failure = probability >= decision_threshold
    delta_t = reading.process_temperature - reading.air_temperature
    power_watts = reading.torque * reading.rotational_speed * (2.0 * math.pi / 60.0)

    return PredictionResponse(
        id=f"PRED-{uuid.uuid4().hex[:8].upper()}",
        sample_id=f"SAMPLE-{uuid.uuid4().hex[:4].upper()}",
        prediction_code=1 if is_failure else 0,
        prediction_label=(
            "Failure Predicted (Action Required)" if is_failure else "Normal Operation"
        ),
        failure_probability=round(probability, 4),
        failure_percentage=percentage,
        decision_threshold=decision_threshold,
        risk_level=level,
        risk_tier=tier,
        recommended_action=action,
        contributing_factors=physical_contributing_factors(reading),
        sensors=SensorsOut(
            air_temperature=reading.air_temperature,
            process_temperature=reading.process_temperature,
            rotational_speed=reading.rotational_speed,
            torque=reading.torque,
            tool_wear=reading.tool_wear,
            delta_t=round(delta_t, 1),
            power_watts=round(power_watts),
        ),
        model_version=MODEL_VERSION,
        model_provenance={
            "model_name": predictor.model_name,
            "algorithm": predictor.metadata.get("algorithm_class", "Classifier"),
            "training_dataset": "AI4I 2020 Predictive Maintenance Dataset",
        },
        explanation_source="physics-heuristics",
        is_demo=False,
        source="sklearn-random-forest",
        timestamp=datetime.now(timezone.utc).isoformat(),
        disclaimer=raw.get(
            "disclaimer",
            "Statistical risk classification only; not a guaranteed time-to-failure.",
        ),
    )


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------


def _cors_origins() -> list[str]:
    raw = os.getenv(
        "MACHINEGUARD_CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,"
        "http://localhost:4173,http://127.0.0.1:4173",
    )
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


@asynccontextmanager
async def lifespan(_: FastAPI):
    get_predictor()  # Fail fast if the model is missing or unreadable.
    yield
    logger.info("Shutting down MachineGuard API")


app = FastAPI(
    title="MachineGuard Inference API",
    description=(
        "Serves the trained AI4I 2020 Random Forest failure classifier. "
        "Inputs are the five physical sensor channels; outputs include a "
        "continuous failure probability, risk tier and decision."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/", tags=["system"])
def index() -> dict[str, Any]:
    return {
        "service": "MachineGuard Inference API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "health": "/health",
            "metadata": "/api/model/metadata",
            "predict": "/api/predict",
            "batch_predict": "/api/predict/batch",
        },
    }


@app.get("/health", tags=["system"])
def health() -> dict[str, Any]:
    try:
        predictor = get_predictor()
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=503, detail=f"Model unavailable: {exc}") from exc

    return {
        "status": "ok",
        "model_loaded": True,
        "model_name": predictor.model_name,
        "decision_threshold": predictor.threshold,
        "model_version": MODEL_VERSION,
        "uptime_seconds": round(time.time() - SERVICE_STARTED_AT, 1),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/model/metadata", tags=["model"])
def model_metadata() -> dict[str, Any]:
    predictor = get_predictor()
    metadata = predictor.metadata

    return {
        "model_name": metadata.get("model_name"),
        "algorithm": metadata.get("algorithm_class"),
        "model_version": MODEL_VERSION,
        "decision_threshold": predictor.threshold,
        "primary_features": metadata.get("primary_features", []),
        "engineered_features": metadata.get("internal_derived_features", []),
        "all_features": metadata.get("total_internal_features", []),
        "feature_importance": feature_importance(predictor),
        "test_metrics": metadata.get("test_metrics", {}),
        "validation_metrics": metadata.get("validation_metrics", {}),
        "confusion_matrix": metadata.get("confusion_matrix", {}),
        "hyperparameters": metadata.get("hyperparameters", {}),
        "training_data_info": metadata.get("training_data_info", {}),
        "disclaimer": metadata.get("disclaimer"),
    }


@app.post("/api/predict", response_model=PredictionResponse, tags=["prediction"])
def predict(request: PredictRequest) -> PredictionResponse:
    predictor = get_predictor()
    try:
        return build_prediction(predictor, request, request.threshold)
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc


@app.post(
    "/api/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["prediction"],
)
def predict_batch(request: BatchPredictRequest) -> BatchPredictionResponse:
    predictor = get_predictor()
    try:
        results = [
            build_prediction(predictor, reading, request.threshold)
            for reading in request.readings
        ]
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Batch prediction failed")
        raise HTTPException(
            status_code=500, detail=f"Batch prediction failed: {exc}"
        ) from exc

    return BatchPredictionResponse(
        count=len(results),
        model_version=MODEL_VERSION,
        results=results,
    )
