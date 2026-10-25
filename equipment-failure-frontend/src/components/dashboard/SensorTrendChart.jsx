import React, { useState } from 'react';
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid
} from 'recharts';
import { TELEMETRY_TIME_SERIES } from '../../data/demoDashboardData';

const METRICS = [
  { key: 'failureProbability', label: 'Failure Prob.', unit: '%', color: '#EF4444', min: 0, max: 100 },
  { key: 'torque', label: 'Torque', unit: 'Nm', color: '#2563EB', min: 30, max: 75 },
  { key: 'rotationalSpeed', label: 'Spindle Speed', unit: 'rpm', color: '#0D9488', min: 1200, max: 1600 },
  { key: 'toolWear', label: 'Tool Wear', unit: 'min', color: '#D97706', min: 0, max: 260 },
  { key: 'processTemp', label: 'Process Temp', unit: 'K', color: '#EA580C', min: 305, max: 316 }
];

// Clean Daylight Tooltip
function CustomTooltip({ active, payload, label }) {
  if (active && payload && payload.length) {
    const data = payload[0].payload;
    return (
      <div className="custom-recharts-tooltip">
        <div className="tooltip-header">Timestamp: {label}</div>
        <div className="tooltip-row">
          <span><span className="tooltip-dot" style={{ background: '#EF4444' }} />Failure Risk:</span>
          <b className="mono">{data.failureProbability}%</b>
        </div>
        <div className="tooltip-row">
          <span><span className="tooltip-dot" style={{ background: '#2563EB' }} />Torque:</span>
          <b className="mono">{data.torque} Nm</b>
        </div>
        <div className="tooltip-row">
          <span><span className="tooltip-dot" style={{ background: '#0D9488' }} />Speed:</span>
          <b className="mono">{data.rotationalSpeed} rpm</b>
        </div>
        <div className="tooltip-row">
          <span><span className="tooltip-dot" style={{ background: '#D97706' }} />Tool Wear:</span>
          <b className="mono">{data.toolWear} min</b>
        </div>
        <div className="tooltip-row">
          <span><span className="tooltip-dot" style={{ background: '#EA580C' }} />Process Temp:</span>
          <b className="mono">{data.processTemp} K</b>
        </div>
      </div>
    );
  }
  return null;
}

export default function SensorTrendChart() {
  const [selectedMetric, setSelectedMetric] = useState('failureProbability');

  const activeConfig = METRICS.find(m => m.key === selectedMetric) || METRICS[0];

  return (
    <div className="card">
      <div className="card-header" style={{ alignItems: 'center' }}>
        <div className="card-title-group">
          <h3 className="card-title">24-Hour Telemetry & Risk Trajectory</h3>
          <span className="card-subtitle">Continuous time-series sensor readings across industrial shift</span>
        </div>

        {/* Clean Segmented Control Tabs (No wrapping) */}
        <div className="segmented-control" role="tablist" aria-label="Metric selectors">
          {METRICS.map(m => (
            <button
              key={m.key}
              type="button"
              role="tab"
              aria-selected={selectedMetric === m.key}
              className={`segmented-tab ${selectedMetric === m.key ? 'active' : ''}`}
              onClick={() => setSelectedMetric(m.key)}
            >
              {m.label}
            </button>
          ))}
        </div>
      </div>

      <div style={{ width: '100%', height: 280, marginTop: '0.75rem' }}>
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={TELEMETRY_TIME_SERIES} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
            <defs>
              <linearGradient id={`gradient-${activeConfig.key}`} x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor={activeConfig.color} stopOpacity={0.2} />
                <stop offset="95%" stopColor={activeConfig.color} stopOpacity={0.0} />
              </linearGradient>
            </defs>
            <CartesianGrid stroke="#E2E8F0" strokeDasharray="3 3" vertical={false} />
            <XAxis
              dataKey="time"
              tick={{ fill: '#64748B', fontSize: 12, fontFamily: 'var(--font-mono)' }}
              axisLine={{ stroke: '#E2E8F0' }}
              tickLine={false}
            />
            <YAxis
              domain={[activeConfig.min, activeConfig.max]}
              tick={{ fill: '#64748B', fontSize: 12, fontFamily: 'var(--font-mono)' }}
              axisLine={false}
              tickLine={false}
              unit={` ${activeConfig.unit}`}
            />
            <Tooltip content={<CustomTooltip />} />
            <Area
              type="monotone"
              dataKey={activeConfig.key}
              stroke={activeConfig.color}
              strokeWidth={2}
              fillOpacity={1}
              fill={`url(#gradient-${activeConfig.key})`}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
