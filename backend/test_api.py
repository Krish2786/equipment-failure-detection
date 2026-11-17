"""
Smoke tests for the MachineGuard inference API.

Run from the project root:
    python -m pytest backend/test_api.py -q

Requires: pytest (or run `python backend/test_api.py` for the built-in runner).
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from backend.main import MODEL_VERSION, app  # noqa: E402

client = TestClient(app)

NOMINAL = {
    "airTemperature": 298.2,
    "processTemperature": 308.7,
    "rotationalSpeed": 1500,
    "torque": 40.0,
    "toolWear": 35,
}

OVERSTRAIN = {
    "airTemperature": 301.5,
    "processTemperature": 310.8,
    "rotationalSpeed": 1300,
    "torque": 65.5,
    "toolWear": 215,
}


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_metadata() -> None:
    response = client.get("/api/model/metadata")
    assert response.status_code == 200
    body = response.json()
    assert body["model_version"] == MODEL_VERSION
    assert body["decision_threshold"] == 0.35
    assert len(body["all_features"]) == 8
    assert len(body["feature_importance"]) == 8


def test_predict_nominal() -> None:
    response = client.post("/api/predict", json=NOMINAL)
    assert response.status_code == 200
    body = response.json()
    assert body["isDemo"] is False
    assert body["source"] == "sklearn-random-forest"
    assert body["failureProbability"] < 0.35
    assert body["predictionCode"] == 0
    assert body["sensors"]["deltaT"] == 10.5
    assert len(body["contributingFactors"]) >= 1


def test_predict_overstrain_flags_failure() -> None:
    response = client.post("/api/predict", json=OVERSTRAIN)
    assert response.status_code == 200
    body = response.json()
    assert body["failureProbability"] >= 0.35
    assert body["predictionCode"] == 1
    assert body["riskTier"] in {"High Risk", "Critical Risk"}
    assert any("overstrain" in factor.lower() for factor in body["contributingFactors"])


def test_snake_case_input_accepted() -> None:
    payload = {
        "air_temperature": 298.2,
        "process_temperature": 308.7,
        "rotational_speed": 1500,
        "torque": 40.0,
        "tool_wear": 35,
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200


def test_threshold_override() -> None:
    response = client.post("/api/predict", json={**NOMINAL, "threshold": 0.01})
    assert response.status_code == 200
    assert response.json()["decisionThreshold"] == 0.01


def test_rejects_negative_thermal_gradient() -> None:
    payload = {**NOMINAL, "processTemperature": 295.0}
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 422


def test_batch() -> None:
    response = client.post(
        "/api/predict/batch",
        json={"readings": [NOMINAL, OVERSTRAIN]},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 2
    assert body["results"][0]["predictionCode"] == 0
    assert body["results"][1]["predictionCode"] == 1


if __name__ == "__main__":
    failures = 0
    for name, case in sorted(globals().items()):
        if name.startswith("test_") and callable(case):
            try:
                case()
                print(f"PASS {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}: {exc}")
    raise SystemExit(1 if failures else 0)
