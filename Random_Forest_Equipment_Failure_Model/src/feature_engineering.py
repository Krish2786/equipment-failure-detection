"""
Feature Engineering Module for Equipment Failure Detection
===========================================================
Derives 3 domain-physics features from the 5 primary machine telemetry sensor inputs:
1. delta_temperature [K]: Temperature gradient between milling process and ambient air
2. mechanical_power [W]: Mechanical power delivered to the spindle (Torque * Rotational speed * 2*pi / 60)
3. overstrain_index [Nm * min]: Mechanical stress accumulation (Torque * Tool wear)
"""

import numpy as np
import pandas as pd

# The 5 raw sensor inputs provided by factory machinery / frontend
PRIMARY_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

# The 3 physics-derived feature names
ENGINEERED_FEATURE_NAMES = [
    "delta_temperature",
    "mechanical_power",
    "overstrain_index"
]

# Full 8-feature representation fed into the Random Forest classifier
ALL_8_FEATURES = PRIMARY_FEATURES + ENGINEERED_FEATURE_NAMES

TARGET = "Machine failure"

def convert_to_numpy(X):
    """
    Transforms DataFrame or array-like to NumPy ndarray.
    Used by Scikit-learn pipelines to standardize array structures.
    """
    if isinstance(X, pd.DataFrame):
        return X.to_numpy()
    return np.asarray(X)

def add_physics_features(X):
    """
    Derives 3 physics-based features from the 5 primary sensor features:
    1. delta_temperature = Process temperature - Air temperature [K]
    2. mechanical_power = Torque * Rotational speed * 2 * pi / 60 [Watts]
    3. overstrain_index = Torque * Tool wear [Nm * min]

    Accepts:
    - pd.DataFrame with PRIMARY_FEATURES columns
    - or 2D np.ndarray of shape (N, 5) with columns in standard order:
      [Air temp, Process temp, Speed, Torque, Tool wear]

    Returns:
    - 2D np.ndarray of shape (N, 8) containing all 8 features in order:
      [5 raw features, delta_temperature, mechanical_power, overstrain_index]
    """
    if isinstance(X, pd.DataFrame):
        air_t = X["Air temperature [K]"].to_numpy(dtype=float)
        proc_t = X["Process temperature [K]"].to_numpy(dtype=float)
        speed = X["Rotational speed [rpm]"].to_numpy(dtype=float)
        torque = X["Torque [Nm]"].to_numpy(dtype=float)
        tool_w = X["Tool wear [min]"].to_numpy(dtype=float)
    else:
        arr = np.asarray(X, dtype=float)
        air_t = arr[:, 0]
        proc_t = arr[:, 1]
        speed = arr[:, 2]
        torque = arr[:, 3]
        tool_w = arr[:, 4]

    # Calculate physics metrics
    delta_t = proc_t - air_t
    mech_power = torque * speed * 2.0 * np.pi / 60.0
    overstrain = torque * tool_w

    if isinstance(X, pd.DataFrame):
        base_arr = X[PRIMARY_FEATURES].to_numpy(dtype=float)
    else:
        base_arr = np.asarray(X, dtype=float)[:, :5]

    return np.column_stack([base_arr, delta_t, mech_power, overstrain])
