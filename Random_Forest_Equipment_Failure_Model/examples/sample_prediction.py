"""
Sample Prediction Script
========================
Demonstrates loading test telemetry from JSON and evaluating failure risk
using the optimized Random Forest predictive maintenance pipeline.
"""

import sys
import json
from pathlib import Path

# Setup paths to import from src/
examples_dir = Path(__file__).resolve().parent
pkg_root = examples_dir.parent
src_dir = pkg_root / "src"
for p in [str(src_dir), str(pkg_root)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.model_prediction import EquipmentFailurePredictor


def run_sample_predictions():
    print("=" * 80)
    print("RUNNING EQUIPMENT FAILURE DETECTION PREDICTION DEMO")
    print("=" * 80)

    # 1. Initialize predictor
    predictor = EquipmentFailurePredictor()

    # 2. Load sample inputs
    json_path = examples_dir / "sample_input.json"
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    scenarios = data.get("test_scenarios", [])
    print(f"Loaded {len(scenarios)} test scenarios from: {json_path.name}\n")

    for sc in scenarios:
        sid = sc["scenario_id"]
        desc = sc["description"]
        telem = sc["telemetry"]
        expected = sc.get("expected_outcome", "N/A")

        print("-" * 80)
        print(f"[{sid}] {desc}")
        print(f"  Inputs: AirTemp={telem['air_temp']}K, ProcTemp={telem['process_temp']}K, Speed={telem['rotational_speed']}rpm, Torque={telem['torque']}Nm, ToolWear={telem['tool_wear']}min")
        print(f"  Expected: {expected}")

        # Run prediction
        res = predictor.predict(
            air_temperature=telem["air_temp"],
            process_temperature=telem["process_temp"],
            rotational_speed=telem["rotational_speed"],
            torque=telem["torque"],
            tool_wear=telem["tool_wear"]
        )

        code = res["prediction_code"]
        label = res["prediction_label"]
        prob = res["failure_probability"]
        risk = res["risk_level"]
        thresh = res["decision_threshold"]
        action = res["recommended_action"]

        print(f"\n  >> PREDICTION RESULT:")
        print(f"     Status:           {label} (Code: {code})")
        print(f"     Failure Prob:     {prob*100:.2f}% (Threshold: {thresh})")
        print(f"     Risk Level:       {risk}")
        print(f"     Action Required:  {action}")

    print("\n" + "=" * 80)
    print("SAMPLE PREDICTIONS COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    run_sample_predictions()
