# MachineGuard Inference API

FastAPI service that loads the trained Random Forest
(`ml/models/best_model/ai4i_failure_model.joblib`) and exposes real model
predictions to the React dashboard.

## Setup

```bash
# From the project root
python -m pip install -r backend/requirements.txt
```

## Run

```bash
# From the project root
python -m uvicorn backend.main:app --reload --port 8000

# Or from this directory
cd backend
python -m uvicorn main:app --reload --port 8000
```

Interactive API docs: <http://localhost:8000/docs>

## Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness and model status |
| `GET` | `/api/model/metadata` | Model card, metrics, feature importance |
| `POST` | `/api/predict` | Score one sensor reading |
| `POST` | `/api/predict/batch` | Score up to 5,000 readings |

### Predict

```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "airTemperature": 301.5,
    "processTemperature": 310.8,
    "rotationalSpeed": 1300,
    "torque": 65.5,
    "toolWear": 215
  }'
```

Both `camelCase` and `snake_case` input keys are accepted. The response uses
`camelCase` so it can be consumed directly by the frontend.

Example response:

```json
{
  "id": "PRED-1A2B3C4D",
  "predictionCode": 1,
  "predictionLabel": "Failure Predicted (Action Required)",
  "failureProbability": 0.9641,
  "failurePercentage": 96.4,
  "decisionThreshold": 0.35,
  "riskLevel": "critical",
  "riskTier": "Critical Risk",
  "recommendedAction": "Immediate intervention recommended. Halt cycle or dispatch maintenance engineer immediately.",
  "contributingFactors": ["..."],
  "sensors": { "airTemperature": 301.5, "deltaT": 9.3, "powerWatts": 8919, "...": "..." },
  "modelVersion": "rf-ai4i-optimized-v1",
  "isDemo": false,
  "source": "sklearn-random-forest",
  "timestamp": "2026-11-09T12:00:00+00:00"
}
```

### Batch

```bash
curl -X POST http://localhost:8000/api/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"readings":[{"airTemperature":298.4,"processTemperature":308.7,"rotationalSpeed":1500,"torque":40.2,"toolWear":120}]}'
```

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `MACHINEGUARD_CORS_ORIGINS` | `http://localhost:5173,...` | Comma-separated allowed origins |
| `MACHINEGUARD_MODEL_VERSION` | `rf-ai4i-optimized-v1` | Version string returned to the UI |
| `MACHINEGUARD_LOG_LEVEL` | `INFO` | Python log level |

## Connect the React app

In `equipment-failure-frontend/.env`:

```bash
VITE_API_URL="http://localhost:8000/api"
VITE_USE_MOCK_PREDICTOR="false"
```

Then restart `npm run dev`. With the flag omitted (or `"true"`) the dashboard
uses its standalone physics heuristic so the demo still works without Python.

## Notes

- The model pipeline references `ml/src/utils.py`, which the service adds to
  `sys.path` before loading the joblib file.
- The shipped model was serialized with scikit-learn 1.9.1. Loading it with a
  different version emits `InconsistentVersionWarning`; predictions still work,
  but pinning the training version is recommended for production.
- `contributingFactors` are physics-threshold summaries, not SHAP values.
  Per-prediction SHAP explanation is a natural next increment.
