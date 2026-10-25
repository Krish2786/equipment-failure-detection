/**
 * Sensor boundary definitions & validation logic for industrial equipment telemetry.
 */

export const SENSOR_BOUNDS = {
  airTemperature: {
    key: 'airTemperature',
    label: 'Air Temperature',
    unit: 'K',
    min: 290.0,
    max: 310.0,
    step: 0.1,
    defaultVal: 298.4,
    description: 'Ambient facility temperature surrounding the equipment enclosure'
  },
  processTemperature: {
    key: 'processTemperature',
    label: 'Process Temperature',
    unit: 'K',
    min: 300.0,
    max: 320.0,
    step: 0.1,
    defaultVal: 308.7,
    description: 'Generated internal mechanical process temperature'
  },
  rotationalSpeed: {
    key: 'rotationalSpeed',
    label: 'Rotational Speed',
    unit: 'rpm',
    min: 1000,
    max: 3000,
    step: 10,
    defaultVal: 1500,
    description: 'Spindle rotational velocity'
  },
  torque: {
    key: 'torque',
    label: 'Torque',
    unit: 'Nm',
    min: 3.0,
    max: 80.0,
    step: 0.1,
    defaultVal: 40.2,
    description: 'Rotational mechanical torque exerted by spindle'
  },
  toolWear: {
    key: 'toolWear',
    label: 'Tool Wear',
    unit: 'min',
    min: 0,
    max: 260,
    step: 1,
    defaultVal: 120,
    description: 'Cumulative cutting tool engagement time'
  }
};

/**
 * Validate an individual sensor input against defined physical bounds.
 * @param {string} sensorKey
 * @param {number|string} value
 * @returns {string|null} Error message or null if valid
 */
export function validateSensor(sensorKey, value) {
  const spec = SENSOR_BOUNDS[sensorKey];
  if (!spec) return 'Unknown sensor parameter';

  if (value === '' || value === null || value === undefined) {
    return `${spec.label} is required`;
  }

  const num = Number(value);
  if (isNaN(num)) {
    return `${spec.label} must be a valid number`;
  }

  if (num < spec.min) {
    return `${spec.label} cannot be below ${spec.min} ${spec.unit}`;
  }

  if (num > spec.max) {
    return `${spec.label} cannot exceed ${spec.max} ${spec.unit}`;
  }

  return null;
}

/**
 * Validates a set of 5 sensor readings.
 * @param {Object} sensorValues
 * @returns {{ isValid: boolean, errors: Object }}
 */
export function validateSensorReadings(sensorValues) {
  const errors = {};
  let isValid = true;

  for (const key of Object.keys(SENSOR_BOUNDS)) {
    const error = validateSensor(key, sensorValues[key]);
    if (error) {
      errors[key] = error;
      isValid = false;
    }
  }

  // Cross-sensor boundary check: Process temp should generally be >= Air Temp in active operation
  if (!errors.airTemperature && !errors.processTemperature) {
    const air = Number(sensorValues.airTemperature);
    const proc = Number(sensorValues.processTemperature);
    if (proc < air) {
      errors.processTemperature = 'Process Temperature should not be lower than Air Temperature during active operation';
      isValid = false;
    }
  }

  return { isValid, errors };
}
