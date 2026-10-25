import React from 'react';

export default function SensorInputField({
  spec,
  value,
  onChange,
  error
}) {
  const handleInputChange = (e) => {
    const val = e.target.value;
    onChange(spec.key, val === '' ? '' : Number(val));
  };

  const handleSliderChange = (e) => {
    onChange(spec.key, Number(e.target.value));
  };

  return (
    <div className="sensor-field-wrapper">
      <div className="sensor-field-header">
        <div className="sensor-field-label-group">
          <label htmlFor={`sensor-${spec.key}`} className="sensor-field-name">
            {spec.label}
          </label>
          <span className="sensor-field-unit">{spec.unit}</span>
        </div>

        <div className="sensor-field-input-box">
          <input
            id={`sensor-${spec.key}`}
            type="number"
            className="sensor-number-input"
            value={value !== undefined ? value : ''}
            min={spec.min}
            max={spec.max}
            step={spec.step}
            onChange={handleInputChange}
          />
        </div>
      </div>

      <div className="sensor-slider-box">
        <input
          type="range"
          className="sensor-range-slider"
          min={spec.min}
          max={spec.max}
          step={spec.step}
          value={value !== undefined ? value : spec.defaultVal}
          onChange={handleSliderChange}
          aria-label={`${spec.label} slider`}
        />
        <div className="sensor-bounds-display">
          <span>Min: {spec.min} {spec.unit}</span>
          <span>Nominal: {spec.defaultVal} {spec.unit}</span>
          <span>Max: {spec.max} {spec.unit}</span>
        </div>
      </div>

      {spec.description && (
        <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
          {spec.description}
        </span>
      )}

      {error && <span className="sensor-field-error">{error}</span>}
    </div>
  );
}
