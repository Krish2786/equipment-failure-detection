import React from 'react';
import { getRiskTierDetails } from '../../utils/formatters';

/**
 * Renders an operational risk tier badge mapped from failure percentage or tier string.
 * Operational Risk Tiers:
 * - Optimal Operation (< 25%)
 * - Elevated Wear (25% - 49.9%)
 * - High Failure Risk (50% - 74.9%)
 * - Critical Imminent Risk (>= 75%)
 */
export default function RiskBadge({ probability, tier, showPercentage = false, className = '' }) {
  let details;

  if (probability !== undefined && probability !== null) {
    details = getRiskTierDetails(probability);
  } else if (tier) {
    // Lookup by tier name
    const lower = tier.toLowerCase();
    if (lower.includes('optimal') || lower.includes('low')) {
      details = { tier: 'Optimal Operation', badgeClass: 'tier-optimal' };
    } else if (lower.includes('elevated') || lower.includes('wear') || lower.includes('moderate')) {
      details = { tier: 'Elevated Wear', badgeClass: 'tier-moderate' };
    } else if (lower.includes('high')) {
      details = { tier: 'High Failure Risk', badgeClass: 'tier-high' };
    } else {
      details = { tier: 'Critical Imminent Risk', badgeClass: 'tier-critical' };
    }
  } else {
    details = { tier: 'Unknown', badgeClass: 'tier-optimal' };
  }

  return (
    <span className={`risk-badge ${details.badgeClass} ${className}`}>
      <span className="risk-badge-dot" />
      <span>{details.tier}</span>
      {showPercentage && probability !== undefined && (
        <span className="mono" style={{ marginLeft: '0.25rem', opacity: 0.9 }}>
          ({Number(probability).toFixed(1)}%)
        </span>
      )}
    </span>
  );
}
