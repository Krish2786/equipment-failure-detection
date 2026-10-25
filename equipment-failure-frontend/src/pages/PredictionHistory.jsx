import React, { useState, useMemo } from 'react';
import {
  Download,
  Trash2,
  Search,
  RotateCcw,
  Eye,
  X,
  History,
  AlertTriangle,
  Cpu
} from 'lucide-react';
import { usePredictionContext } from '../context/PredictionContext';
import { formatPercentage, formatTimestamp, formatSensorValue, getRiskTierDetails } from '../utils/formatters';
import PageHeader from '../components/layout/PageHeader';
import RiskBadge from '../components/common/RiskBadge';
import EmptyState from '../components/common/EmptyState';
import ConfirmDialog from '../components/common/ConfirmDialog';

export default function PredictionHistory() {
  const { predictions, clearPredictions, resetToDemoPredictions } = usePredictionContext();

  const [searchTerm, setSearchTerm] = useState('');
  const [selectedTier, setSelectedTier] = useState('ALL');
  const [sortOrder, setSortOrder] = useState('desc'); // 'desc' | 'asc'
  const [showClearConfirm, setShowClearConfirm] = useState(false);
  const [activeModalRecord, setActiveModalRecord] = useState(null);

  // Filter and sort predictions
  const filteredPredictions = useMemo(() => {
    return predictions
      .filter((p) => {
        const matchesSearch =
          p.id.toLowerCase().includes(searchTerm.toLowerCase()) ||
          (p.sampleId && p.sampleId.toLowerCase().includes(searchTerm.toLowerCase())) ||
          (p.equipmentName && p.equipmentName.toLowerCase().includes(searchTerm.toLowerCase()));

        const matchesTier =
          selectedTier === 'ALL' ||
          (p.riskTier && p.riskTier.toLowerCase().includes(selectedTier.toLowerCase()));

        return matchesSearch && matchesTier;
      })
      .sort((a, b) => {
        const dateA = new Date(a.timestamp).getTime();
        const dateB = new Date(b.timestamp).getTime();
        return sortOrder === 'desc' ? dateB - dateA : dateA - dateB;
      });
  }, [predictions, searchTerm, selectedTier, sortOrder]);

  // Export history to CSV
  const handleExportCSV = () => {
    const headers = [
      'Prediction ID',
      'Sample ID',
      'Equipment Name',
      'Timestamp',
      'Air Temperature (K)',
      'Process Temperature (K)',
      'Rotational Speed (rpm)',
      'Torque (Nm)',
      'Tool Wear (min)',
      'Failure Probability (%)',
      'Risk Tier',
      'Model Version',
      'Recommended Action'
    ];

    const rows = filteredPredictions.map((p) => {
      const s = p.sensors || {};
      const probPct = p.failurePercentage || (p.failureProbability * 100);
      return [
        `"${p.id}"`,
        `"${p.sampleId || ''}"`,
        `"${p.equipmentName || ''}"`,
        `"${p.timestamp}"`,
        s.airTemperature || '',
        s.processTemperature || '',
        s.rotationalSpeed || '',
        s.torque || '',
        s.toolWear || '',
        probPct,
        `"${p.riskTier}"`,
        `"${p.modelVersion || 'rf-sklearn-demo-v1'}"`,
        `"${(p.recommendedAction || '').replace(/"/g, '""')}"`
      ];
    });

    const csvContent = 'data:text/csv;charset=utf-8,' + [headers.join(','), ...rows.map(e => e.join(','))].join('\n');
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', `machineguard_predictions_history_${new Date().toISOString().slice(0, 10)}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const getBarColor = (pct) => {
    if (pct < 25) return 'var(--risk-optimal)';
    if (pct < 50) return 'var(--risk-moderate)';
    if (pct < 75) return 'var(--risk-high)';
    return 'var(--risk-critical)';
  };

  return (
    <div className="fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      <PageHeader
        title="Prediction History & Audit Log"
        description="Inspect all telemetry diagnostic runs recorded during this session alongside baseline historical records. Click any row to review full physical telemetry."
        actions={
          <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
            {predictions.length === 0 ? (
              <button className="btn-secondary btn-sm" onClick={resetToDemoPredictions}>
                <RotateCcw size={14} />
                <span>Reload Demo History</span>
              </button>
            ) : (
              <>
                <button
                  className="btn-outline btn-sm"
                  onClick={() => setShowClearConfirm(true)}
                  style={{ color: 'var(--risk-critical)', borderColor: 'var(--risk-critical-border)' }}
                >
                  <Trash2 size={14} />
                  <span>Clear History</span>
                </button>
                <button className="btn-primary btn-sm" onClick={handleExportCSV}>
                  <Download size={14} />
                  <span>Export History CSV</span>
                </button>
              </>
            )}
          </div>
        }
      />

      {/* Filter and Search Bar */}
      <div className="card" style={{ padding: '1.25rem' }}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '1rem', alignItems: 'center' }}>
          {/* Search */}
          <div style={{ position: 'relative' }}>
            <Search size={16} style={{ position: 'absolute', left: 10, top: '50%', transform: 'translateY(-50%)', color: 'var(--text-muted)' }} />
            <input
              type="text"
              placeholder="Search by PRED ID or Sample..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              style={{ width: '100%', paddingLeft: '2.2rem' }}
            />
          </div>

          {/* Risk Tier Filter */}
          <select
            value={selectedTier}
            onChange={(e) => setSelectedTier(e.target.value)}
          >
            <option value="ALL">All Risk Tiers</option>
            <option value="Optimal">Optimal Operation (&lt;25%)</option>
            <option value="Elevated">Elevated Wear (25-49.9%)</option>
            <option value="High">High Failure Risk (50-74.9%)</option>
            <option value="Critical">Critical Imminent Risk (&gt;=75%)</option>
          </select>

          {/* Sort Order */}
          <select
            value={sortOrder}
            onChange={(e) => setSortOrder(e.target.value)}
          >
            <option value="desc">Sort by Date: Newest First</option>
            <option value="asc">Sort by Date: Oldest First</option>
          </select>
        </div>
      </div>

      {/* Prediction History Table */}
      <div className="card">
        {filteredPredictions.length === 0 ? (
          <EmptyState
            icon={History}
            title="No Prediction Records Found"
            description={predictions.length === 0 ? "You have cleared your prediction history. You can reload the demonstration dataset or run a new diagnostic evaluation." : "No records match the current filter criteria."}
            actionLabel={predictions.length === 0 ? "Reload Demo Records" : "Reset Filter"}
            onAction={predictions.length === 0 ? resetToDemoPredictions : () => { setSearchTerm(''); setSelectedTier('ALL'); }}
          />
        ) : (
          <div className="table-container">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Prediction ID</th>
                  <th>Equipment / Sample</th>
                  <th>Timestamp</th>
                  <th>Sensor Summary</th>
                  <th>Continuous Probability</th>
                  <th>Risk Tier</th>
                  <th style={{ textAlign: 'center' }}>Action</th>
                </tr>
              </thead>
              <tbody>
                {filteredPredictions.map((p) => {
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
                      <td style={{ textAlign: 'center' }}>
                        <button
                          className="btn-outline btn-sm"
                          onClick={() => setActiveModalRecord(p)}
                          title="Inspect complete telemetry and diagnostic notes"
                        >
                          <Eye size={14} />
                          <span>View Details</span>
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Detail Inspection Modal Dialog */}
      {activeModalRecord && (
        <div className="modal-overlay" onClick={() => setActiveModalRecord(null)}>
          <div className="modal-dialog" onClick={(e) => e.stopPropagation()} style={{ maxWidth: 600 }}>
            <div className="modal-header">
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem' }}>
                <Cpu size={20} style={{ color: 'var(--accent-primary)' }} />
                <h3 className="modal-title">Evaluation Detail: {activeModalRecord.id}</h3>
              </div>
              <button
                className="btn-outline btn-sm btn-icon-only"
                onClick={() => setActiveModalRecord(null)}
                aria-label="Close dialog"
              >
                <X size={18} />
              </button>
            </div>

            {/* Probability Banner */}
            <div style={{ padding: '1rem', background: 'var(--bg-elevated)', borderRadius: 'var(--radius-md)', display: 'flex', alignItems: 'center', justifyContent: 'space-between', border: '1px solid var(--border-color)' }}>
              <div>
                <span style={{ fontSize: '0.72rem', textTransform: 'uppercase', color: 'var(--text-muted)', fontWeight: 600 }}>
                  Failure Probability
                </span>
                <div className="mono" style={{ fontSize: '1.8rem', fontWeight: 800, color: getRiskTierDetails(activeModalRecord.failurePercentage || activeModalRecord.failureProbability * 100).color }}>
                  {formatPercentage(activeModalRecord.failurePercentage || activeModalRecord.failureProbability * 100)}
                </div>
              </div>
              <RiskBadge probability={activeModalRecord.failurePercentage || activeModalRecord.failureProbability * 100} />
            </div>

            {/* Sensor Inputs Grid */}
            <div>
              <div style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: '0.5rem' }}>
                Physical Sensor Telemetry:
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '0.5rem', fontSize: '0.82rem' }}>
                <div style={{ background: 'var(--bg-elevated)', padding: '0.6rem 0.8rem', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-color)' }}>
                  <span style={{ color: 'var(--text-muted)' }}>Air Temperature:</span>
                  <div className="mono" style={{ fontWeight: 600 }}>{formatSensorValue('airTemp', activeModalRecord.sensors?.airTemperature)}</div>
                </div>
                <div style={{ background: 'var(--bg-elevated)', padding: '0.6rem 0.8rem', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-color)' }}>
                  <span style={{ color: 'var(--text-muted)' }}>Process Temperature:</span>
                  <div className="mono" style={{ fontWeight: 600 }}>{formatSensorValue('processTemp', activeModalRecord.sensors?.processTemperature)}</div>
                </div>
                <div style={{ background: 'var(--bg-elevated)', padding: '0.6rem 0.8rem', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-color)' }}>
                  <span style={{ color: 'var(--text-muted)' }}>Rotational Speed:</span>
                  <div className="mono" style={{ fontWeight: 600 }}>{formatSensorValue('rotationalSpeed', activeModalRecord.sensors?.rotationalSpeed)}</div>
                </div>
                <div style={{ background: 'var(--bg-elevated)', padding: '0.6rem 0.8rem', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-color)' }}>
                  <span style={{ color: 'var(--text-muted)' }}>Torque:</span>
                  <div className="mono" style={{ fontWeight: 600 }}>{formatSensorValue('torque', activeModalRecord.sensors?.torque)}</div>
                </div>
                <div style={{ background: 'var(--bg-elevated)', padding: '0.6rem 0.8rem', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-color)', gridColumn: 'span 2' }}>
                  <span style={{ color: 'var(--text-muted)' }}>Tool Wear (engagement):</span>
                  <div className="mono" style={{ fontWeight: 600 }}>{formatSensorValue('toolWear', activeModalRecord.sensors?.toolWear)}</div>
                </div>
              </div>
            </div>

            {/* Contributing factors */}
            <div>
              <div style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: '0.5rem' }}>
                Contributing Physical Factors:
              </div>
              <div className="contributing-factors-list" style={{ marginTop: 0 }}>
                {activeModalRecord.contributingFactors?.map((f, i) => (
                  <div key={i} className="factor-item" style={{ padding: '0.5rem 0.75rem' }}>
                    <span className="factor-bullet" />
                    <span>{f}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Recommended Action */}
            <div className="recommendation-box" style={{ margin: 0 }}>
              <div className="rec-title">
                <AlertTriangle size={15} />
                <span>Operational Recommendation</span>
              </div>
              <p className="rec-text">{activeModalRecord.recommendedAction}</p>
            </div>

            <div className="modal-actions">
              <button className="btn-secondary" onClick={() => setActiveModalRecord(null)}>
                Close
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Confirmation Dialog for Clear History */}
      <ConfirmDialog
        isOpen={showClearConfirm}
        title="Clear Prediction History?"
        message="This will remove all recorded session evaluations and demo predictions from your browser session storage. You can reload the demonstration dataset at any time."
        confirmLabel="Clear All Records"
        isDestructive={true}
        onConfirm={() => {
          clearPredictions();
          setShowClearConfirm(false);
        }}
        onCancel={() => setShowClearConfirm(false)}
      />
    </div>
  );
}
