import React from 'react';
import { Activity, CheckCircle, AlertTriangle, XCircle, Clock } from 'lucide-react';

export default function EquipmentStatusCard({ records = [] }) {
  const total = records.length || 40;
  const activeCount = records.filter(r => r.status === 'Active').length || 28;
  const maintCount = records.filter(r => r.status === 'Maintenance Due').length || 5;
  const warnCount = records.filter(r => r.status === 'Warning').length || 4;
  const critCount = records.filter(r => r.status === 'Critical Alert').length || 3;

  const activePct = Math.round((activeCount / total) * 100);
  const maintPct = Math.round((maintCount / total) * 100);
  const warnPct = Math.round((warnCount / total) * 100);
  const critPct = Math.round((critCount / total) * 100);

  return (
    <div className="card">
      <div className="card-header">
        <div className="card-title-group">
          <h3 className="card-title">
            <Activity size={18} style={{ color: 'var(--accent-primary)' }} />
            <span>Industrial Fleet Operational Status</span>
          </h3>
          <span className="card-subtitle">Real-time status of {total} connected production units</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <span className="risk-badge tier-optimal">
            <span className="risk-badge-dot" />
            <span>{activePct}% Fleet Health Index</span>
          </span>
        </div>
      </div>

      {/* Segmented bar */}
      <div style={{ height: '8px', display: 'flex', borderRadius: 'var(--radius-full)', overflow: 'hidden', margin: '0.85rem 0', background: '#F1F5F9' }}>
        <div style={{ width: `${activePct}%`, background: 'var(--risk-optimal)' }} title={`Active: ${activeCount}`} />
        <div style={{ width: `${maintPct}%`, background: 'var(--risk-moderate)' }} title={`Maintenance Due: ${maintCount}`} />
        <div style={{ width: `${warnPct}%`, background: 'var(--risk-high)' }} title={`Warning: ${warnCount}`} />
        <div style={{ width: `${critPct}%`, background: 'var(--risk-critical)' }} title={`Critical Alert: ${critCount}`} />
      </div>

      {/* Breakdown list */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: '0.75rem', marginTop: '0.75rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '13px', color: 'var(--text-secondary)' }}>
          <CheckCircle size={15} style={{ color: 'var(--risk-optimal)' }} />
          <span>Active: <b className="mono" style={{ color: 'var(--text-primary)' }}>{activeCount}</b></span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '13px', color: 'var(--text-secondary)' }}>
          <Clock size={15} style={{ color: 'var(--risk-moderate)' }} />
          <span>Maint. Due: <b className="mono" style={{ color: 'var(--text-primary)' }}>{maintCount}</b></span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '13px', color: 'var(--text-secondary)' }}>
          <AlertTriangle size={15} style={{ color: 'var(--risk-high)' }} />
          <span>Warning: <b className="mono" style={{ color: 'var(--text-primary)' }}>{warnCount}</b></span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '13px', color: 'var(--text-secondary)' }}>
          <XCircle size={15} style={{ color: 'var(--risk-critical)' }} />
          <span>Critical: <b className="mono" style={{ color: 'var(--text-primary)' }}>{critCount}</b></span>
        </div>
      </div>
    </div>
  );
}
