import React from 'react';
import { Loader2 } from 'lucide-react';

export default function LoadingState({ message = 'Evaluating industrial telemetry streams...' }) {
  return (
    <div className="state-container fade-in">
      <div className="state-icon-box" style={{ borderColor: 'var(--accent-primary)' }}>
        <Loader2 size={30} style={{ animation: 'spin 1s linear infinite', color: 'var(--accent-primary)' }} />
      </div>
      <div className="state-title" style={{ color: 'var(--accent-primary)' }}>
        Processing Diagnostic Model
      </div>
      <div className="state-desc mono" style={{ fontSize: '0.8rem' }}>
        {message}
      </div>
    </div>
  );
}
