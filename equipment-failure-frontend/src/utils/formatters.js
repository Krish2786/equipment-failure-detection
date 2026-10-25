/**
 * Formatting utilities for industrial telemetry and ML outputs.
 */

/**
 * Format a probability value (0-1 or 0-100) into a clean percentage string.
 * @param {number} value - Probability between 0 and 1, or percentage between 0 and 100
 * @param {number} decimals - Number of decimal places (default: 1)
 * @returns {string} Formatted percentage (e.g. "14.2%")
 */
export function formatPercentage(value, decimals = 1) {
  if (value === null || value === undefined || isNaN(value)) {
    return '0.0%';
  }
  // If value is between 0 and 1, treat as probability fraction
  const pct = value <= 1 && value >= 0 ? value * 100 : value;
  return `${pct.toFixed(decimals)}%`;
}

/**
 * Format a raw ISO timestamp or date into an industrial telemetry timestamp.
 * @param {string|Date} dateInput
 * @returns {string} Formatted date (e.g. "2026-10-25 14:30:15")
 */
export function formatTimestamp(dateInput) {
  if (!dateInput) return '—';
  try {
    const d = new Date(dateInput);
    if (isNaN(d.getTime())) return String(dateInput);
    
    const pad = (n) => String(n).padStart(2, '0');
    const year = d.getFullYear();
    const month = pad(d.getMonth() + 1);
    const day = pad(d.getDate());
    const hours = pad(d.getHours());
    const minutes = pad(d.getMinutes());
    const seconds = pad(d.getSeconds());
    
    return `${year}-${month}-${day} ${hours}:${minutes}:${seconds}`;
  } catch (err) {
    return String(dateInput);
  }
}

/**
 * Formats sensor values with their respective engineering units.
 */
export function formatSensorValue(name, val) {
  if (val === null || val === undefined || isNaN(val)) return '—';
  const num = Number(val);
  switch (name) {
    case 'airTemp':
    case 'airTemperature':
      return `${num.toFixed(1)} K`;
    case 'processTemp':
    case 'processTemperature':
      return `${num.toFixed(1)} K`;
    case 'rotationalSpeed':
      return `${Math.round(num).toLocaleString()} rpm`;
    case 'torque':
      return `${num.toFixed(1)} Nm`;
    case 'toolWear':
      return `${Math.round(num)} min`;
    default:
      return String(num);
  }
}

/**
 * Determine the Operational Risk Tier from a continuous failure probability percentage.
 * Risk Tiers:
 * - 0.0% – 24.9%: Optimal Operation
 * - 25.0% – 49.9%: Elevated Wear
 * - 50.0% – 74.9%: High Failure Risk
 * - 75.0% – 100.0%: Critical Imminent Risk
 * 
 * @param {number} probabilityPct - Percentage (0-100)
 * @returns {{ tier: string, level: string, color: string, action: string }}
 */
export function getRiskTierDetails(probabilityPct) {
  const pct = Number(probabilityPct) || 0;

  if (pct < 25.0) {
    return {
      tier: 'Optimal Operation',
      level: 'optimal',
      color: 'var(--risk-optimal)',
      badgeClass: 'tier-optimal',
      action: 'Routine monitoring active. No intervention required.'
    };
  } else if (pct < 50.0) {
    return {
      tier: 'Elevated Wear',
      level: 'moderate',
      color: 'var(--risk-moderate)',
      badgeClass: 'tier-moderate',
      action: 'Schedule physical inspection during the next planned maintenance window.'
    };
  } else if (pct < 75.0) {
    return {
      tier: 'High Failure Risk',
      level: 'high',
      color: 'var(--risk-high)',
      badgeClass: 'tier-high',
      action: 'Prioritize preventive maintenance. Inspect cutting tool, bearings, and thermal loops.'
    };
  } else {
    return {
      tier: 'Critical Imminent Risk',
      level: 'critical',
      color: 'var(--risk-critical)',
      badgeClass: 'tier-critical',
      action: 'Immediate intervention recommended. Halt cycle or dispatch maintenance engineer immediately.'
    };
  }
}
