import React from 'react';
import { useNavigate } from 'react-router-dom';
import { ArrowUpRight, PlusCircle, Clock } from 'lucide-react';
import { usePredictionContext } from '../../context/PredictionContext';
import { formatTimestamp, formatPercentage } from '../../utils/formatters';
import RiskBadge from '../common/RiskBadge';

export default function RecentPredictionsTable() {
  const navigate = useNavigate();
  const { stats } = usePredictionContext();
  const predictions = stats.recentPredictions || [];

  const getBarColor = (pct) => {
    if (pct < 25) return 'var(--risk-optimal)';
    if (pct < 50) return 'var(--risk-moderate)';
    if (pct < 75) return 'var(--risk-high)';
    return 'var(--risk-critical)';
  };

  return (
    <div className="card">
      <div className="card-header">
        <div className="card-title-group">
          <h3 className="card-title">
            <Clock size={18} style={{ color: 'var(--accent-primary)' }} />
            <span>Recent Diagnostic Evaluations</span>
          </h3>
          <span className="card-subtitle">Real-time telemetry evaluations processed by the Random Forest probability model</span>
        </div>

        <div style={{ display: 'flex', gap: '0.5rem' }}>
          <button className="btn-secondary btn-sm" onClick={() => navigate('/history')}>
            <span>View All History</span>
            <ArrowUpRight size={14} />
          </button>
          <button className="btn-primary btn-sm" onClick={() => navigate('/predict')}>
            <PlusCircle size={14} />
            <span>Run New Diagnostic</span>
          </button>
        </div>
      </div>

      <div className="table-container">
        <table className="data-table">
          <thead>
            <tr>
              <th>Evaluation ID</th>
              <th>Sample / Equipment</th>
              <th>Timestamp</th>
              <th>Sensor Parameters</th>
              <th>Failure Probability (%)</th>
              <th>Operational Risk Tier</th>
            </tr>
          </thead>
          <tbody>
            {predictions.length === 0 ? (
              <tr>
                <td colSpan={6} style={{ textAlign: 'center', padding: '2rem', color: 'var(--text-muted)' }}>
                  No recent predictions recorded yet. Run a diagnostic evaluation to see live results.
                </td>
              </tr>
            ) : (
              predictions.map((p) => {
                const probPct = p.failurePercentage || (p.failureProbability * 100);
                const s = p.sensors || {};
                return (
                  <tr key={p.id}>
                    <td>
                      <span className="mono" style={{ fontWeight: 600, color: 'var(--accent-primary)' }}>
                        {p.id}
                      </span>
                    </td>
                    <td>
                      <div style={{ display: 'flex', flexDirection: 'column' }}>
                        <span style={{ fontWeight: 600, color: 'var(--text-primary)' }}>
                          {p.equipmentName || p.sampleId || 'CNC Spindle'}
                        </span>
                        <span className="mono" style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
                          {p.sampleId}
                        </span>
                      </div>
                    </td>
                    <td>
                      <span className="mono" style={{ fontSize: '0.78rem' }}>
                        {formatTimestamp(p.timestamp)}
                      </span>
                    </td>
                    <td>
                      <span className="mono" style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                        {s.airTemperature ? `${s.airTemperature}K | ${s.rotationalSpeed}rpm | ${s.torque}Nm | ${s.toolWear}m` : 'Telemetry within bounds'}
                      </span>
                    </td>
                    <td>
                      <div className="prob-bar-wrapper">
                        <div className="prob-bar-track">
                          <div
                            className="prob-bar-fill"
                            style={{
                              width: `${Math.min(probPct, 100)}%`,
                              backgroundColor: getBarColor(probPct)
                            }}
                          />
                        </div>
                        <span className="prob-bar-text" style={{ color: getBarColor(probPct) }}>
                          {formatPercentage(probPct)}
                        </span>
                      </div>
                    </td>
                    <td>
                      <RiskBadge probability={probPct} />
                    </td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
