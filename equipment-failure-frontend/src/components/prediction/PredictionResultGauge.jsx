import React from 'react';
import { Cpu, AlertCircle, CheckCircle, Info, Copy, Check } from 'lucide-react';
import { formatPercentage, formatTimestamp, getRiskTierDetails } from '../../utils/formatters';
import RiskBadge from '../common/RiskBadge';
import LoadingState from '../common/LoadingState';

export default function PredictionResultGauge({ result, isEvaluating, error }) {
  const [copied, setCopied] = React.useState(false);

  if (isEvaluating) {
    return (
      <div className="card card-elevated" style={{ height: '100%', minHeight: 480, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <LoadingState message="Extracting features • Running Random Forest continuous probability estimator..." />
      </div>
    );
  }

  if (error) {
    return (
      <div className="card card-elevated" style={{ height: '100%', minHeight: 480, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', textAlign: 'center', padding: '2rem' }}>
        <AlertCircle size={40} style={{ color: 'var(--risk-critical)', marginBottom: '1rem' }} />
        <h4 style={{ color: 'var(--risk-critical)', marginBottom: '0.5rem' }}>Evaluation Failed</h4>
        <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', maxWidth: 360 }}>{error}</p>
      </div>
    );
  }

  if (!result) {
    return (
      <div className="card card-elevated" style={{ height: '100%', minHeight: 480, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', textAlign: 'center', padding: '2.5rem' }}>
        <div className="state-icon-box" style={{ width: 64, height: 64, marginBottom: '1rem' }}>
          <Cpu size={32} style={{ color: 'var(--color-primary)' }} />
        </div>
        <h3 style={{ fontSize: '1.2rem', marginBottom: '0.5rem' }}>Ready for Diagnostic Evaluation</h3>
        <p style={{ color: 'var(--text-muted)', fontSize: '0.88rem', maxWidth: 400, lineHeight: 1.6 }}>
          Configure the 5 physical sensor inputs or choose an engineering preset on the left, then click <b>Calculate Failure Probability</b> to generate a continuous ML risk assessment.
        </p>
      </div>
    );
  }

  const probPct = result.failurePercentage !== undefined ? result.failurePercentage : (result.failureProbability * 100);
  const tierDetails = getRiskTierDetails(probPct);

  // Radial Gauge Math
  // Gauge is a semi-circle from 180 deg to 360 deg
  // Radius = 80, Circumference = Math.PI * 80 = 251.3
  const radius = 80;
  const circumference = Math.PI * radius;
  const strokeDashoffset = circumference - (circumference * Math.min(probPct, 100)) / 100;

  const handleCopy = () => {
    const text = `MachineGuard Diagnostic Report
ID: ${result.id}
Failure Probability: ${formatPercentage(probPct)}
Risk Tier: ${tierDetails.tier}
Model Version: ${result.modelVersion}
Timestamp: ${result.timestamp}
Contributing Factors:
${result.contributingFactors.map(f => `- ${f}`).join('\n')}
Action: ${result.recommendedAction}`;
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="card card-elevated fade-in" style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <div className="card-header">
        <div className="card-title-group">
          <h3 className="card-title">
            <Cpu size={18} style={{ color: 'var(--accent-primary)' }} />
            <span>Diagnostic Inference Result</span>
          </h3>
          <span className="card-subtitle">
            Evaluation ID: <span className="mono" style={{ color: 'var(--accent-primary)' }}>{result.id}</span>
          </span>
        </div>

        <button
          className="btn-outline btn-sm"
          onClick={handleCopy}
          title="Copy diagnostic report to clipboard"
        >
          {copied ? <Check size={14} style={{ color: 'var(--risk-optimal)' }} /> : <Copy size={14} />}
          <span>{copied ? 'Copied' : 'Copy Report'}</span>
        </button>
      </div>

      {/* Radial Probability Gauge */}
      <div className="gauge-container">
        <div className="gauge-svg-wrapper">
          <svg width="220" height="135" viewBox="0 0 220 135">
            {/* Background Track Arc */}
            <path
              d="M 30 115 A 80 80 0 0 1 190 115"
              fill="none"
              stroke="#E2E8F0"
              strokeWidth="16"
              strokeLinecap="round"
            />
            {/* Value Arc */}
            <path
              d="M 30 115 A 80 80 0 0 1 190 115"
              fill="none"
              stroke={tierDetails.color}
              strokeWidth="16"
              strokeLinecap="round"
              strokeDasharray={circumference}
              strokeDashoffset={strokeDashoffset}
              style={{ transition: 'stroke-dashoffset 0.6s cubic-bezier(0.16, 1, 0.3, 1), stroke 0.3s ease' }}
            />
          </svg>

          <div className="gauge-readout-center">
            <div className="gauge-percentage-value" style={{ color: tierDetails.color }}>
              {formatPercentage(probPct)}
            </div>
            <div className="gauge-percentage-label">Failure Probability</div>
          </div>
        </div>

        <div style={{ marginTop: '0.75rem', display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <RiskBadge probability={probPct} />
          <span className="mono" style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
            v: {result.modelVersion}
          </span>
        </div>
      </div>

      {/* Action Recommendation Box */}
      <div className="recommendation-box" style={{ borderLeftColor: tierDetails.color }}>
        <div className="rec-title" style={{ color: tierDetails.color }}>
          <AlertCircle size={15} />
          <span>Recommended Engineering Action</span>
        </div>
        <p className="rec-text">{result.recommendedAction}</p>
      </div>

      {/* Dynamic Contributing Factors Breakdown */}
      <div style={{ marginTop: '1.25rem', flex: 1 }}>
        <div style={{ fontSize: '0.8rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>
          Contributing Telemetry Stress Factors:
        </div>
        <div className="contributing-factors-list">
          {result.contributingFactors && result.contributingFactors.map((factor, index) => (
            <div key={index} className="factor-item">
              <span className="factor-bullet" style={{ background: tierDetails.color }} />
              <span style={{ color: 'var(--text-secondary)' }}>{factor}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Timestamp & Verification Footer */}
      <div style={{ marginTop: '1.5rem', paddingTop: '0.75rem', borderTop: '1px solid var(--border-color)', display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '0.72rem', color: 'var(--text-muted)' }}>
        <span className="mono">Timestamp: {formatTimestamp(result.timestamp)}</span>
        <span style={{ display: 'flex', alignItems: 'center', gap: '0.35rem', color: 'var(--risk-optimal)' }}>
          <CheckCircle size={12} />
          <span>Persisted to Session History</span>
        </span>
      </div>
    </div>
  );
}
