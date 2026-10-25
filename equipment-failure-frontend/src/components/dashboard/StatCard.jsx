import React from 'react';

export default function StatCard({
  label,
  value,
  icon: Icon,
  trend,
  trendDirection = 'neutral', // 'positive' | 'negative' | 'neutral'
  subtext,
  className = ''
}) {
  return (
    <div className={`card stat-card ${className}`}>
      <div className="stat-card-top">
        <span className="stat-card-label">{label}</span>
        {Icon && (
          <div className="stat-card-icon-wrap">
            <Icon size={18} />
          </div>
        )}
      </div>

      <div className="stat-card-value">{value}</div>

      {(trend || subtext) && (
        <div className="stat-card-bottom">
          {trend && (
            <span className={`stat-card-trend trend-${trendDirection}`}>
              {trend}
            </span>
          )}
          {subtext && <span>{subtext}</span>}
        </div>
      )}
    </div>
  );
}
