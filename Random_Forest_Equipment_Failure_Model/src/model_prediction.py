"""
AI4I 2020 Predictive Maintenance - Model Prediction & Inference Interface
==========================================================================
Loads the serialized Random Forest production model and metadata to evaluate
equipment failure risk based on 5 real-time continuous sensor telemetry inputs:
- Air temperature [K]
- Process temperature [K]
- Rotational speed [rpm]
- Torque [Nm]
- Tool wear [min]

Usage:
    from src.model_prediction import predict_failure_risk
    result = predict_failure_risk(
        air_temp=301.5,
        process_temp=310.8,
        rotational_speed=1300,
        torque=65.5,
        tool_wear=215
    )
    print(result)
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, Union

# Ensure current and parent directories are in sys.path
src_dir = Path(__file__).resolve().parent
pkg_root = src_dir.parent
for p in [str(src_dir), str(pkg_root)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Local imports
try:
    from .feature_engineering import PRIMARY_FEATURES, add_physics_features
    from . import feature_engineering as fe
except ImportError:
    from feature_engineering import PRIMARY_FEATURES, add_physics_features
    import feature_engineering as fe

# Register 'utils' alias in sys.modules so joblib can deserialize FunctionTransformer(add_physics_features)
sys.modules['utils'] = fe

import numpy as np
import pandas as pd
import joblib

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("ai4i_predict")


class EquipmentFailurePredictor:
    """Production predictor wrapper for the optimized AI4I Random Forest model."""

    def __init__(self, model_path: Union[str, Path] = None, metadata_path: Union[str, Path] = None):
        """
        Initializes the predictor by resolving paths and loading the Joblib model.
        """
        # Resolve model path
        if model_path is None:
            candidates = [
                pkg_root / "model" / "ai4i_failure_model.joblib",
                src_dir / "model" / "ai4i_failure_model.joblib",
                Path("model") / "ai4i_failure_model.joblib",
                Path("../model") / "ai4i_failure_model.joblib"
            ]
            model_path = next((c for c in candidates if c.exists()), None)
            if model_path is None:
                raise FileNotFoundError(
                    f"Model binary not found in standard paths: {[str(c) for c in candidates]}"
                )
        else:
            model_path = Path(model_path)

        # Resolve metadata path
        if metadata_path is None:
            meta_candidates = [
                model_path.parent / "model_metadata.json",
                pkg_root / "model" / "model_metadata.json",
                Path("model") / "model_metadata.json"
            ]
            metadata_path = next((c for c in meta_candidates if c.exists()), None)
        else:
            metadata_path = Path(metadata_path)

        self.model_path = model_path
        self.metadata_path = metadata_path

        # Load model binary
        self.model = joblib.load(self.model_path)

        # Load metadata if available
        if self.metadata_path and self.metadata_path.exists():
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}

        self.features = self.metadata.get("primary_features", PRIMARY_FEATURES)
        self.threshold = float(self.metadata.get("selected_probability_threshold", 0.35))
        self.model_name = self.metadata.get("model_name", "Optimized Random Forest")
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
        Generates risk assessment and failure prediction for the 5 raw sensor inputs.

        Parameters:
            air_temperature: Ambient temperature in Kelvin [K]
            process_temperature: Process temperature in Kelvin [K]
            rotational_speed: Spindle rotational speed in revolutions per minute [rpm]
            torque: Spindle mechanical torque in Newton-meters [Nm]
            tool_wear: Cumulative cutting tool wear in minutes [min]

        Returns:
            Dictionary containing prediction code, label, continuous probability,
            operational threshold, risk level, action recommendation, and provenance.
        """
        # Form DataFrame with exact column names in required order
        input_data = pd.DataFrame([{
            "Air temperature [K]": float(air_temperature),
            "Process temperature [K]": float(process_temperature),
            "Rotational speed [rpm]": float(rotational_speed),
            "Torque [Nm]": float(torque),
            "Tool wear [min]": float(tool_wear)
        }])[self.features]

        # Generate continuous failure probability using model.predict_proba()
        if hasattr(self.model, "predict_proba"):
            failure_prob = float(self.model.predict_proba(input_data)[0, 1])
        else:
            pred_raw = int(self.model.predict(input_data)[0])
            failure_prob = 1.0 if pred_raw == 1 else 0.0

        # Apply calibrated operational threshold (0.35)
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

        result = {
            "prediction_code": prediction_code,
            "prediction_label": prediction_label,
            "failure_probability": round(failure_prob, 4),
            "decision_threshold": self.threshold,
            "risk_level": risk_level,
            "recommended_action": recommended_action,
            "telemetry_inputs": {
                "Air temperature [K]": float(air_temperature),
                "Process temperature [K]": float(process_temperature),
                "Rotational speed [rpm]": float(rotational_speed),
                "Torque [Nm]": float(torque),
                "Tool wear [min]": float(tool_wear)
            },
            "model_provenance": {
                "model_name": self.model_name,
                "model_family": self.metadata.get("model_family", "Random Forest"),
                "algorithm": self.metadata.get("algorithm_class", "Pipeline(FunctionTransformer + RandomForestClassifier)"),
                "version": "Phase 2.1 Optimized"
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
    """
    Convenience function for external modules to evaluate failure risk directly.
    """
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
    print("AI4I 2020 RANDOM FOREST MODEL - INFERENCE VERIFICATION")
    print("=" * 80)

    predictor = EquipmentFailurePredictor()

    # Test Scenario 1: Healthy Nominal Milling Operation
    print("\n--- Test Scenario 1: Nominal Machine Telemetry ---")
    case_1 = predictor.predict(
        air_temperature=298.2,
        process_temperature=308.7,
        rotational_speed=1500,
        torque=40.0,
        tool_wear=35
    )
    print(json.dumps(case_1, indent=2))

    # Test Scenario 2: Overstrain Failure Profile (High Torque + Worn Tool)
    print("\n--- Test Scenario 2: Overstrain Failure Profile (High Torque + Worn Tool) ---")
    case_2 = predictor.predict(
        air_temperature=301.5,
        process_temperature=310.8,
        rotational_speed=1300,
        torque=65.5,
        tool_wear=215
    )
    print(json.dumps(case_2, indent=2))

    # Test Scenario 3: Heat Dissipation Anomaly Profile (Low Delta-T + Low Speed)
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
    print("INFERENCE VERIFICATION COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()
