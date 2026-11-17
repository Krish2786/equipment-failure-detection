"""
Compatibility module re-exporting feature engineering functions.
Allows deserialization of models saved with reference to 'utils.add_physics_features'.
"""

from .feature_engineering import (
    PRIMARY_FEATURES,
    ENGINEERED_FEATURE_NAMES,
    ALL_8_FEATURES,
    TARGET,
    convert_to_numpy,
    add_physics_features
)

__all__ = [
    "PRIMARY_FEATURES",
    "ENGINEERED_FEATURE_NAMES",
    "ALL_8_FEATURES",
    "TARGET",
    "convert_to_numpy",
    "add_physics_features"
]
