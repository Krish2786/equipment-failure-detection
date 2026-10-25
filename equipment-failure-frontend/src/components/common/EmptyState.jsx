import React from 'react';
import { AlertCircle } from 'lucide-react';

export default function EmptyState({
  icon: Icon = AlertCircle,
  title = 'No Data Found',
  description = 'No matching telemetry records or prediction results are currently available.',
  actionLabel,
  onAction
}) {
  return (
    <div className="state-container fade-in">
      <div className="state-icon-box">
        <Icon size={28} />
      </div>
      <div className="state-title">{title}</div>
      <div className="state-desc">{description}</div>
      {actionLabel && onAction && (
        <button className="btn-secondary btn-sm" onClick={onAction}>
          {actionLabel}
        </button>
      )}
    </div>
  );
}
