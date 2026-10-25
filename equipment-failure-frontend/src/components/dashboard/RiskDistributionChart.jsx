import React from 'react';
import { ResponsiveContainer, PieChart, Pie, Cell, Tooltip } from 'recharts';
import { usePredictionContext } from '../../context/PredictionContext';

function CustomPieTooltip({ active, payload }) {
  if (active && payload && payload.length) {
    const data = payload[0].payload;
    return (
      <div className="custom-recharts-tooltip">
        <div className="tooltip-header">{data.name}</div>
        <div className="tooltip-row">
          <span>Evaluations:</span>
          <b className="mono">{data.count} units</b>
        </div>
        <div className="tooltip-row">
          <span>Fleet Share:</span>
          <b className="mono">{data.percentage}%</b>
        </div>
      </div>
    );
  }
  return null;
}

export default function RiskDistributionChart() {
  const { stats } = usePredictionContext();
  const data = stats.riskDistribution;

  return (
    <div className="card">
      <div className="card-header">
        <div className="card-title-group">
          <h3 className="card-title">Risk Tier Distribution</h3>
          <span className="card-subtitle">Operational categorization of evaluated telemetry</span>
        </div>
        <span className="mono" style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
          Total: {stats.totalEvaluations}
        </span>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: 210 }}>
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Tooltip content={<CustomPieTooltip />} />
            <Pie
              data={data}
              innerRadius={55}
              outerRadius={85}
              paddingAngle={3}
              dataKey="count"
            >
              {data.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.color} stroke="#FFFFFF" strokeWidth={2} />
              ))}
            </Pie>
          </PieChart>
        </ResponsiveContainer>
      </div>

      {/* Daylight Legend */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '0.65rem', marginTop: '0.5rem' }}>
        {data.map((tier) => (
          <div key={tier.name} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '12px', color: 'var(--text-secondary)' }}>
            <span style={{ width: 8, height: 8, borderRadius: '50%', background: tier.color, flexShrink: 0 }} />
            <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{tier.name}</span>
            <b className="mono" style={{ marginLeft: 'auto', color: 'var(--text-primary)' }}>{tier.count}</b>
          </div>
        ))}
      </div>
    </div>
  );
}
