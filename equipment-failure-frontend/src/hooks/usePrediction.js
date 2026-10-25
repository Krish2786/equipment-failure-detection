import { useState, useCallback } from 'react';
import { usePredictionContext } from '../context/PredictionContext';
import { predictEquipmentFailure } from '../services/predictionService';
import { PRESET_CONFIGURATIONS } from '../data/demoDashboardData';
import { SENSOR_BOUNDS, validateSensorReadings } from '../utils/validation';

/**
 * Custom hook for managing failure prediction form submission,
 * loading state, presets, and error handling.
 */
export function usePrediction() {
  const context = usePredictionContext();

  const [sensorValues, setSensorValues] = useState({
    airTemperature: SENSOR_BOUNDS.airTemperature.defaultVal,
    processTemperature: SENSOR_BOUNDS.processTemperature.defaultVal,
    rotationalSpeed: SENSOR_BOUNDS.rotationalSpeed.defaultVal,
    torque: SENSOR_BOUNDS.torque.defaultVal,
    toolWear: SENSOR_BOUNDS.toolWear.defaultVal
  });

  const [errors, setErrors] = useState({});
  const [activePreset, setActivePreset] = useState(null);
  const [isEvaluating, setIsEvaluating] = useState(false);
  const [lastResult, setLastResult] = useState(null);
  const [evaluationError, setEvaluationError] = useState(null);

  // Update a single sensor value and clear error for that field
  const handleSensorChange = useCallback((sensorKey, val) => {
    setActivePreset(null);
    setSensorValues(prev => ({
      ...prev,
      [sensorKey]: val
    }));
    setErrors(prev => {
      if (prev[sensorKey]) {
        const next = { ...prev };
        delete next[sensorKey];
        return next;
      }
      return prev;
    });
  }, []);

  // Apply a quick-fill preset
  const applyPreset = useCallback((presetKey) => {
    const preset = PRESET_CONFIGURATIONS[presetKey];
    if (preset) {
      setSensorValues({ ...preset.values });
      setActivePreset(presetKey);
      setErrors({});
      setEvaluationError(null);
    }
  }, []);

  // Reset form to default baseline
  const resetForm = useCallback(() => {
    setSensorValues({
      airTemperature: SENSOR_BOUNDS.airTemperature.defaultVal,
      processTemperature: SENSOR_BOUNDS.processTemperature.defaultVal,
      rotationalSpeed: SENSOR_BOUNDS.rotationalSpeed.defaultVal,
      torque: SENSOR_BOUNDS.torque.defaultVal,
      toolWear: SENSOR_BOUNDS.toolWear.defaultVal
    });
    setErrors({});
    setActivePreset(null);
    setEvaluationError(null);
  }, []);

  // Execute diagnostic prediction
  const runPrediction = useCallback(async () => {
    const validation = validateSensorReadings(sensorValues);
    if (!validation.isValid) {
      setErrors(validation.errors);
      return null;
    }

    setIsEvaluating(true);
    setEvaluationError(null);

    try {
      const result = await predictEquipmentFailure(sensorValues);
      setLastResult(result);
      context.addPrediction(result);
      return result;
    } catch (err) {
      setEvaluationError(err.message || 'An error occurred during evaluation');
      return null;
    } finally {
      setIsEvaluating(false);
    }
  }, [sensorValues, context]);

  return {
    ...context,
    sensorValues,
    errors,
    activePreset,
    isEvaluating,
    lastResult,
    evaluationError,
    handleSensorChange,
    applyPreset,
    resetForm,
    runPrediction,
    setLastResult
  };
}
