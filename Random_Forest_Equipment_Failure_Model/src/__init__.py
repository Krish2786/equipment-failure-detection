"""
Equipment Failure Detection - Core Prediction & Feature Engineering Package
"""

import sys
from pathlib import Path

# Ensure src directory is in sys.path and register utils alias for unpickling
src_dir = str(Path(__file__).resolve().parent)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from . import feature_engineering
sys.modules['utils'] = feature_engineering

from .model_prediction import EquipmentFailurePredictor, predict_failure_risk
from .feature_engineering import add_physics_features, PRIMARY_FEATURES, ALL_8_FEATURES

__all__ = [
    "EquipmentFailurePredictor",
    "predict_failure_risk",
    "add_physics_features",
    "PRIMARY_FEATURES",
    "ALL_8_FEATURES"
]
