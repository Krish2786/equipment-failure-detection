import React from 'react';
import { Play, RotateCcw, Sparkles } from 'lucide-react';
import { SENSOR_BOUNDS } from '../../utils/validation';
import { PRESET_CONFIGURATIONS } from '../../data/demoDashboardData';
import SensorInputField from './SensorInputField';

export default function SensorInputForm({
  sensorValues,
  onChange,
  onApplyPreset,
  activePreset,
  onReset,
  onSubmit,
  isEvaluating,
  errors
}) {
  const handleSubmit = (e) => {
    e.preventDefault();
    onSubmit();
  };

  return (
    <form className="card" onSubmit={handleSubmit}>
      <div className="card-header">
        <div className="card-title-group">
          <h3 className="card-title">
            <Sparkles size={18} style={{ color: 'var(--accent-primary)' }} />
            <span>Industrial Sensor Telemetry Inputs</span>
          </h3>
          <span className="card-subtitle">Adjust real-time physical parameters or select an engineering stress scenario</span>
        </div>
      </div>

      {/* Preset Quick-Fill Chips */}
      <div style={{ marginBottom: '1.25rem' }}>
        <div style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '0.5rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
          Quick-Fill Engineering Presets:
        </div>
        <div className="presets-group">
          {Object.keys(PRESET_CONFIGURATIONS).map((key) => {
            const p = PRESET_CONFIGURATIONS[key];
            const isActive = activePreset === key;
            return (
              <button
                type="button"
                key={key}
                className={`preset-chip ${isActive ? 'active' : ''}`}
                onClick={() => onApplyPreset(key)}
                title={p.description}
              >
                <span>{p.label}</span>
                <span className="mono" style={{ fontSize: '0.68rem', opacity: 0.8 }}>({p.badge})</span>
              </button>
            );
          })}
        </div>
      </div>

      {/* The 5 Validated Sensor Inputs */}
      <div className="sensor-form-grid">
        <SensorInputField
          spec={SENSOR_BOUNDS.airTemperature}
          value={sensorValues.airTemperature}
          onChange={onChange}
          error={errors.airTemperature}
        />

        <SensorInputField
          spec={SENSOR_BOUNDS.processTemperature}
          value={sensorValues.processTemperature}
          onChange={onChange}
          error={errors.processTemperature}
        />

        <SensorInputField
          spec={SENSOR_BOUNDS.rotationalSpeed}
          value={sensorValues.rotationalSpeed}
          onChange={onChange}
          error={errors.rotationalSpeed}
        />

        <SensorInputField
          spec={SENSOR_BOUNDS.torque}
          value={sensorValues.torque}
          onChange={onChange}
          error={errors.torque}
        />

        <SensorInputField
          spec={SENSOR_BOUNDS.toolWear}
          value={sensorValues.toolWear}
          onChange={onChange}
          error={errors.toolWear}
        />
      </div>

      {/* Form Action Buttons */}
      <div style={{ display: 'flex', gap: '0.75rem', marginTop: '1.5rem', justifyContent: 'flex-end', flexWrap: 'wrap' }}>
        <button
          type="button"
          className="btn-secondary"
          onClick={onReset}
          disabled={isEvaluating}
        >
          <RotateCcw size={15} />
          <span>Reset Defaults</span>
        </button>

        <button
          type="submit"
          className="btn-primary"
          disabled={isEvaluating}
          style={{ minWidth: '220px' }}
        >
          <Play size={16} fill="currentColor" />
          <span>{isEvaluating ? 'Evaluating Telemetry...' : 'Calculate Failure Probability'}</span>
        </button>
      </div>
    </form>
  );
}
