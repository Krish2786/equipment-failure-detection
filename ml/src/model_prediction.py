"""
AI4I 2020 Predictive Maintenance - Model Prediction & Inference Interface
==========================================================================
Loads the serialized production model and metadata to evaluate equipment failure risk
based on 5 real-time sensor telemetry inputs:
- Air temperature [K]
- Process temperature [K]
- Rotational speed [rpm]
- Torque [Nm]
- Tool wear [min]

Usage:
    from model_prediction import predict_failure_risk
    result = predict_failure_risk(
        air_temp=300.5,
        process_temp=310.2,
        rotational_speed=1500,
        torque=40.0,
        tool_wear=60
    )
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, Union

# Local import support
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import convert_to_numpy, PRIMARY_FEATURES, TARGET

import numpy as np
import pandas as pd
import joblib

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("ai4i_predict")


FAILURE_MODE_LABELS = {
    "HDF": "Heat Dissipation Failure",
    "PWF": "Power Failure",
    "OSF": "Overstrain Failure",
    "TWF": "Tool Wear Failure",
    "RNF": "Random Failure",
}


def diagnose_failure_mode(
    air_temperature: float,
    process_temperature: float,
    rotational_speed: float,
    torque: float,
    tool_wear: float,
) -> Dict[str, Any]:
    """Explain the most likely failure mode from known AI4I physics thresholds."""
    delta_temperature = process_temperature - air_temperature
    mechanical_power = torque * rotational_speed * (2.0 * np.pi / 60.0)
    overstrain_index = torque * tool_wear
    candidates: list[tuple[str, float, str]] = []

    if delta_temperature < 8.6:
        candidates.append((
            "HDF",
            (8.6 - delta_temperature) / 8.6,
            f"Low temperature delta ({delta_temperature:.1f} K) suggests poor heat dissipation.",
        ))
    if mechanical_power > 8500 or (mechanical_power < 2200 and torque > 45):
        candidates.append((
            "PWF",
            max(mechanical_power / 8500, 0.5),
            f"Mechanical load pattern indicates power stress ({mechanical_power:.0f} W).",
        ))
    if overstrain_index > 11000:
        candidates.append((
            "OSF",
            overstrain_index / 11000,
            f"Torque-wear overstrain index is elevated ({overstrain_index:.0f}).",
        ))
    if tool_wear >= 210:
        candidates.append((
            "TWF",
            tool_wear / 210,
            f"Tool wear is beyond the replacement guideline ({tool_wear:.0f} min).",
        ))

    candidates.sort(key=lambda candidate: candidate[1], reverse=True)
    if not candidates:
        return {
            "mode_code": "NONE",
            "mode_label": "No dominant failure mode",
            "reason": "Telemetry does not cross a known HDF, PWF, OSF, or TWF physics threshold.",
            "confidence": "Not applicable",
        }

    mode_code, score, reason = candidates[0]
    return {
        "mode_code": mode_code,
        "mode_label": FAILURE_MODE_LABELS[mode_code],
        "reason": reason,
        "confidence": "Physics-based indicator",
        "alternatives": [FAILURE_MODE_LABELS[code] for code, _, _ in candidates[1:]],
    }


class EquipmentFailurePredictor:
    """Production predictor wrapper for AI4I equipment failure model."""

    def __init__(self, model_dir: Union[str, Path] = None):
        if model_dir is None:
            current_dir = Path.cwd().resolve()
            src_dir = Path(__file__).resolve().parent
            candidates = [
                current_dir / "ml" / "models" / "best_model",
                current_dir / "models" / "best_model",
                src_dir.parent / "models" / "best_model",
                src_dir.parent.parent / "ml" / "models" / "best_model",
                src_dir.parent.parent / "Random_Forest_Equipment_Failure_Model" / "model",
            ]
            model_dir = next(
                (p for p in candidates if (p / "ai4i_failure_model.joblib").exists()),
                candidates[0],
            )
        else:
            model_dir = Path(model_dir)

        self.model_dir = model_dir
        self.model_path = model_dir / "ai4i_failure_model.joblib"
        self.metadata_path = model_dir / "model_metadata.json"

        if not self.model_path.exists():
            raise FileNotFoundError(f"Trained model not found at: {self.model_path}")
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Model metadata not found at: {self.metadata_path}")

        # Load model and metadata
        self.model = joblib.load(self.model_path)
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.features = self.metadata.get("primary_features", PRIMARY_FEATURES)
        self.threshold = float(self.metadata.get("selected_probability_threshold", 0.50))
        self.model_name = self.metadata.get("model_name", "AI4I Classifier")
        logger.info(f"Loaded '{self.model_name}' with decision threshold = {self.threshold:.2f}")

    def predict(
        self,
        air_temperature: float,
        process_temperature: float,
        rotational_speed: float,
        torque: float,
        tool_wear: float
    ) -> Dict[str, Any]:
        """
        Generates risk assessment and failure prediction for given sensor values.
        """
        # Form DataFrame with exact column names in required order
        input_data = pd.DataFrame([{
            "Air temperature [K]": float(air_temperature),
            "Process temperature [K]": float(process_temperature),
            "Rotational speed [rpm]": float(rotational_speed),
            "Torque [Nm]": float(torque),
            "Tool wear [min]": float(tool_wear)
        }])[self.features]

        # Generate continuous probability
        if hasattr(self.model, "predict_proba"):
            failure_prob = float(self.model.predict_proba(input_data)[0, 1])
        else:
            pred_raw = int(self.model.predict(input_data)[0])
            failure_prob = 1.0 if pred_raw == 1 else 0.0

        # Apply calibrated operational threshold
        is_failure = bool(failure_prob >= self.threshold)
        prediction_code = 1 if is_failure else 0
        prediction_label = "Failure Predicted (Action Required)" if is_failure else "Normal Operation"

        # Categorize risk severity level
        if failure_prob >= 0.70:
            risk_level = "CRITICAL"
            recommended_action = "Immediate spindle shutdown and comprehensive mechanical inspection required."
        elif failure_prob >= self.threshold:
            risk_level = "HIGH"
            recommended_action = "Schedule preventive maintenance inspection before next production batch."
        elif failure_prob >= 0.20:
            risk_level = "MODERATE"
            recommended_action = "Telemetry elevated; increase sensor monitoring frequency."
        else:
            risk_level = "LOW"
            recommended_action = "Standard operational parameters; proceed with scheduled milling."

        failure_mode = diagnose_failure_mode(
            air_temperature,
            process_temperature,
            rotational_speed,
            torque,
            tool_wear,
        )

        result = {
            "prediction_code": prediction_code,
            "prediction_label": prediction_label,
            "failure_probability": round(failure_prob, 4),
            "decision_threshold": self.threshold,
            "risk_level": risk_level,
            "recommended_action": recommended_action,
            "failure_mode": failure_mode,
            "telemetry_inputs": {
                "Air temperature [K]": float(air_temperature),
                "Process temperature [K]": float(process_temperature),
                "Rotational speed [rpm]": float(rotational_speed),
                "Torque [Nm]": float(torque),
                "Tool wear [min]": float(tool_wear)
            },
            "model_provenance": {
                "model_name": self.model_name,
                "algorithm": self.metadata.get("algorithm_class", "Classifier")
            },
            "disclaimer": (
                "Notice: Statistical risk classification derived from the AI4I 2020 predictive maintenance dataset. "
                "Output probability reflects mathematical deviation toward known failure conditions and does not "
                "represent an absolute deterministic guarantee of future equipment breakdown time."
            )
        }
        return result

def predict_failure_risk(
    air_temp: float,
    process_temp: float,
    rotational_speed: float,
    torque: float,
    tool_wear: float
) -> Dict[str, Any]:
    """Convenience function for external modules."""
    predictor = EquipmentFailurePredictor()
    return predictor.predict(
        air_temperature=air_temp,
        process_temperature=process_temp,
        rotational_speed=rotational_speed,
        torque=torque,
        tool_wear=tool_wear
    )

def main():
    print("=" * 80)
    print("AI4I 2020 PREDICTIVE MAINTENANCE - MODEL INFERENCE TEST")
    print("=" * 80)

    try:
        predictor = EquipmentFailurePredictor()
    except Exception as e:
        print(f"Error initializing predictor: {e}", file=sys.stderr)
        sys.exit(1)

    # Test Case 1: Healthy Nominal Milling Operation
    print("\n--- Test Scenario 1: Nominal Machine Telemetry ---")
    case_1 = predictor.predict(
        air_temperature=298.2,
        process_temperature=308.7,
        rotational_speed=1500,
        torque=40.0,
        tool_wear=35
    )
    print(json.dumps(case_1, indent=2))

    # Test Case 2: Extreme Torque & High Tool Wear (Overstrain Warning)
    print("\n--- Test Scenario 2: Overstrain Failure Profile (High Torque + Worn Tool) ---")
    case_2 = predictor.predict(
        air_temperature=301.5,
        process_temperature=310.8,
        rotational_speed=1300,
        torque=65.5,
        tool_wear=215
    )
    print(json.dumps(case_2, indent=2))

    # Test Case 3: Thermal Dissipation Anomaly (Low Delta-T + Low Speed)
    print("\n--- Test Scenario 3: Heat Dissipation Failure Profile (Thermal Collapse) ---")
    case_3 = predictor.predict(
        air_temperature=303.8,
        process_temperature=311.2,
        rotational_speed=1320,
        torque=52.0,
        tool_wear=140
    )
    print(json.dumps(case_3, indent=2))

    print("\n" + "=" * 80)
    print("INFERENCE TESTING COMPLETED SUCCESSFULLY")
    print("=" * 80)

if __name__ == "__main__":
    main()
