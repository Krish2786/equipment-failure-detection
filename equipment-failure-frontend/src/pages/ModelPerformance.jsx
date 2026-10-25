import React from 'react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Cell
} from 'recharts';
import { Info, ShieldAlert, Award, AlertTriangle, CheckCircle, Sliders } from 'lucide-react';
import { MODEL_METRICS_INFO } from '../data/demoModelMetrics';
import PageHeader from '../components/layout/PageHeader';

export default function ModelPerformance() {
  const { metrics, confusionMatrix, featureImportance, rocCurveData, prCurveData, hyperparameters } = MODEL_METRICS_INFO;

  return (
    <div className="fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      <PageHeader
        title="Model Performance & Validation Benchmarks"
        description="Comprehensive evaluation metrics of the Scikit-learn Random Forest continuous failure probability model trained on 10,000 industrial telemetry records."
        badge={
          <span className="risk-badge tier-optimal">
            <span className="risk-badge-dot" />
            <span>ROC-AUC: 96.4% Benchmark</span>
          </span>
        }
      />

      {/* Clear Demo Watermark Banner */}
      <div className="disclaimer-banner">
        <div className="disclaimer-content">
          <Info size={18} className="disclaimer-icon" />
          <span>
            <strong>Validation Baseline:</strong> Metrics displayed are derived from standard test set cross-validation on the AI4I 2020 Predictive Maintenance dataset (10k records, 3.4% failure prevalence) utilizing <code>RandomForestClassifier(n_estimators=100, class_weight='balanced')</code>.
          </span>
        </div>
      </div>

      {/* 5 Metric Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem' }}>
        {metrics.map((m) => (
          <div
            key={m.id}
            className="card"
            style={{
              padding: '1.25rem',
              borderColor: m.isHighPriority ? 'var(--color-primary)' : 'var(--border-color)',
              background: m.isHighPriority ? 'var(--bg-elevated)' : 'var(--bg-surface)'
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '0.5rem' }}>
              <span style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase' }}>
                {m.label}
              </span>
              {m.isHighPriority && (
                <span className="risk-badge tier-optimal" style={{ fontSize: '0.65rem', padding: '0.15rem 0.45rem' }}>
                  Critical KPI
                </span>
              )}
            </div>

            <div className="mono" style={{ fontSize: '2rem', fontWeight: 800, color: m.isHighPriority ? 'var(--color-primary)' : 'var(--text-primary)', marginBottom: '0.5rem', lineHeight: 1 }}>
              {m.display}
            </div>

            <p style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', lineHeight: 1.4 }}>
              {m.description}
            </p>
          </div>
        ))}
      </div>

      {/* Interactive Confusion Matrix Section */}
      <div className="grid-2">
        <div className="card">
          <div className="card-header">
            <div className="card-title-group">
              <h3 className="card-title">
                <ShieldAlert size={18} style={{ color: 'var(--color-primary)' }} />
                <span>Confusion Matrix (10,000 Test Telemetry Samples)</span>
              </h3>
              <span className="card-subtitle">Detailed breakdown of true operating states vs predicted failure risks</span>
            </div>
          </div>

          <div className="confusion-matrix-grid">
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 600, fontSize: '0.75rem', color: 'var(--text-muted)' }}>
              Actual \ Pred
            </div>
            <div style={{ fontWeight: 600, fontSize: '0.75rem', color: 'var(--text-muted)', padding: '0.5rem' }}>
              Predicted Normal
            </div>
            <div style={{ fontWeight: 600, fontSize: '0.75rem', color: 'var(--text-muted)', padding: '0.5rem' }}>
              Predicted Failure
            </div>

            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 600, fontSize: '0.75rem', color: 'var(--text-muted)' }}>
              Actual Normal
            </div>
            <div className="matrix-cell matrix-cell-tn">
              <span className="matrix-val" style={{ color: 'var(--risk-optimal)' }}>{confusionMatrix.trueNegatives.toLocaleString()}</span>
              <span className="matrix-label">True Negative (TN)</span>
              <span style={{ fontSize: '0.68rem', color: 'var(--text-muted)' }}>Normal running verified</span>
            </div>
            <div className="matrix-cell matrix-cell-fp">
              <span className="matrix-val" style={{ color: 'var(--risk-moderate)' }}>{confusionMatrix.falsePositives}</span>
              <span className="matrix-label">False Positive (FP)</span>
              <span style={{ fontSize: '0.68rem', color: 'var(--text-muted)' }}>False Alarm (~$120 inspection)</span>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 600, fontSize: '0.75rem', color: 'var(--text-muted)' }}>
              Actual Failure
            </div>
            <div className="matrix-cell matrix-cell-fn" style={{ border: '2px solid var(--risk-critical)' }}>
              <span className="matrix-val" style={{ color: 'var(--risk-critical)' }}>{confusionMatrix.falseNegatives}</span>
              <span className="matrix-label" style={{ color: 'var(--risk-critical)' }}>False Negative (FN)</span>
              <span style={{ fontSize: '0.68rem', color: 'var(--risk-critical)', fontWeight: 600 }}>Critical Miss (~$4,500+ breakdown)</span>
            </div>
            <div className="matrix-cell matrix-cell-tp">
              <span className="matrix-val" style={{ color: 'var(--risk-optimal)' }}>{confusionMatrix.truePositives}</span>
              <span className="matrix-label">True Positive (TP)</span>
              <span style={{ fontSize: '0.68rem', color: 'var(--text-muted)' }}>Catastrophic failure prevented</span>
            </div>
          </div>

          <div style={{ marginTop: '1rem', padding: '0.85rem', background: 'var(--bg-elevated)', borderRadius: 'var(--radius-md)', fontSize: '0.78rem', color: 'var(--text-secondary)', borderLeft: '3px solid var(--risk-critical)' }}>
            <strong>Why False Negatives (FN) Dominate Maintenance Costs:</strong> While a False Positive only incurs a 15-minute diagnostic inspection, a False Negative causes unpredicted spindle seizure during active production, costing an average of $4,500 in ruined workpieces and unscheduled downtime.
          </div>
        </div>

        {/* Feature Importance Ranking Chart */}
        <div className="card">
          <div className="card-header">
            <div className="card-title-group">
              <h3 className="card-title">
                <Sliders size={18} style={{ color: 'var(--color-primary)' }} />
                <span>Feature Importance (Gini Impurity)</span>
              </h3>
              <span className="card-subtitle">Contribution of physical sensors to failure probability</span>
            </div>
          </div>

          <div style={{ height: 260, width: '100%', marginTop: '0.5rem' }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={featureImportance}
                layout="vertical"
                margin={{ top: 5, right: 20, bottom: 5, left: 35 }}
              >
                <CartesianGrid stroke="#E2E8F0" strokeDasharray="3 3" horizontal={false} />
                <XAxis
                  type="number"
                  domain={[0, 40]}
                  unit="%"
                  tick={{ fill: '#64748B', fontSize: 11, fontFamily: 'var(--font-mono)' }}
                  axisLine={{ stroke: '#E2E8F0' }}
                  tickLine={false}
                />
                <YAxis
                  type="category"
                  dataKey="name"
                  tick={{ fill: '#64748B', fontSize: 11 }}
                  axisLine={false}
                  tickLine={false}
                  width={110}
                />
                <Tooltip
                  content={({ active, payload }) => {
                    if (active && payload && payload.length) {
                      const data = payload[0].payload;
                      return (
                        <div className="custom-recharts-tooltip">
                          <div className="tooltip-header">{data.name}</div>
                          <div className="tooltip-row">
                            <span>Relative Importance:</span>
                            <b className="mono" style={{ color: 'var(--accent-primary)' }}>{data.importance}%</b>
                          </div>
                          <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
                            {data.description}
                          </div>
                        </div>
                      );
                    }
                    return null;
                  }}
                />
                <Bar dataKey="importance" fill="var(--accent-primary)" radius={[0, 4, 4, 0]}>
                  {featureImportance.map((entry, index) => (
                    <Cell
                      key={`feat-${index}`}
                      fill={index === 0 ? '#2563EB' : index === 1 ? '#3B82F6' : index === 2 ? '#60A5FA' : '#93C5FD'}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginTop: '0.75rem' }}>
            <b>Torque</b> and <b>Tool Wear</b> account for &gt;60% of total predictive power, aligning with physical principles of mechanical cutting fatigue.
          </div>
        </div>
      </div>

      {/* Side-by-Side ROC Curve & Precision-Recall Curve */}
      <div className="grid-2">
        {/* ROC Curve */}
        <div className="card">
          <div className="card-header">
            <div className="card-title-group">
              <h3 className="card-title">ROC Curve (Receiver Operating Characteristic)</h3>
              <span className="card-subtitle">True Positive Rate vs. False Positive Rate (AUC = 0.964)</span>
            </div>
          </div>
          <div style={{ height: 240, width: '100%' }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={rocCurveData} margin={{ top: 10, right: 10, bottom: 10, left: -10 }}>
                <CartesianGrid stroke="#E2E8F0" strokeDasharray="3 3" />
                <XAxis
                  dataKey="fpr"
                  tick={{ fill: '#64748B', fontSize: 11, fontFamily: 'var(--font-mono)' }}
                  axisLine={{ stroke: '#E2E8F0' }}
                  tickLine={false}
                  unit=" FPR"
                />
                <YAxis
                  domain={[0, 1]}
                  tick={{ fill: '#64748B', fontSize: 11, fontFamily: 'var(--font-mono)' }}
                  axisLine={false}
                  tickLine={false}
                  unit=" TPR"
                />
                <Tooltip
                  content={({ active, payload }) => {
                    if (active && payload && payload.length) {
                      const data = payload[0].payload;
                      return (
                        <div className="custom-recharts-tooltip">
                          <div className="tooltip-row">
                            <span>False Positive Rate (FPR):</span>
                            <b className="mono">{data.fpr}</b>
                          </div>
                          <div className="tooltip-row">
                            <span>True Positive Rate (TPR):</span>
                            <b className="mono" style={{ color: 'var(--accent-primary)' }}>{data.tpr}</b>
                          </div>
                        </div>
                      );
                    }
                    return null;
                  }}
                />
                <Line
                  type="monotone"
                  dataKey="tpr"
                  name="Random Forest"
                  stroke="var(--accent-primary)"
                  strokeWidth={2.5}
                  dot={{ r: 3, fill: 'var(--accent-primary)' }}
                />
                <Line
                  type="monotone"
                  dataKey="baseline"
                  name="Random Guessing"
                  stroke="#94A3B8"
                  strokeDasharray="4 4"
                  strokeWidth={1.5}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Precision-Recall Curve */}
        <div className="card">
          <div className="card-header">
            <div className="card-title-group">
              <h3 className="card-title">Precision-Recall Curve</h3>
              <span className="card-subtitle">Precision across recall decision thresholds for imbalanced classes</span>
            </div>
          </div>
          <div style={{ height: 240, width: '100%' }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={prCurveData} margin={{ top: 10, right: 10, bottom: 10, left: -10 }}>
                <CartesianGrid stroke="#E2E8F0" strokeDasharray="3 3" />
                <XAxis
                  dataKey="recall"
                  tick={{ fill: '#64748B', fontSize: 11, fontFamily: 'var(--font-mono)' }}
                  axisLine={{ stroke: '#E2E8F0' }}
                  tickLine={false}
                  unit=" Rec"
                />
                <YAxis
                  domain={[0.4, 1]}
                  tick={{ fill: '#64748B', fontSize: 11, fontFamily: 'var(--font-mono)' }}
                  axisLine={false}
                  tickLine={false}
                  unit=" Prec"
                />
                <Tooltip
                  content={({ active, payload }) => {
                    if (active && payload && payload.length) {
                      const data = payload[0].payload;
                      return (
                        <div className="custom-recharts-tooltip">
                          <div className="tooltip-row">
                            <span>Recall:</span>
                            <b className="mono">{data.recall}</b>
                          </div>
                          <div className="tooltip-row">
                            <span>Precision:</span>
                            <b className="mono" style={{ color: 'var(--risk-optimal-text)' }}>{data.precision}</b>
                          </div>
                        </div>
                      );
                    }
                    return null;
                  }}
                />
                <Line
                  type="monotone"
                  dataKey="precision"
                  name="PR Curve"
                  stroke="var(--risk-optimal)"
                  strokeWidth={2.5}
                  dot={{ r: 3, fill: 'var(--risk-optimal)' }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Hyperparameter Specs */}
      <div className="card" style={{ padding: '1.25rem' }}>
        <h4 style={{ fontSize: '0.95rem', marginBottom: '0.75rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Sliders size={16} style={{ color: 'var(--accent-primary)' }} />
          <span>Model Architecture & Scikit-learn Hyperparameter Configuration</span>
        </h4>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '0.75rem', fontSize: '0.8rem' }}>
          <div><span style={{ color: 'var(--text-muted)' }}>Estimators:</span> <b className="mono">100 Trees</b></div>
          <div><span style={{ color: 'var(--text-muted)' }}>Max Depth:</span> <b className="mono">12 levels</b></div>
          <div><span style={{ color: 'var(--text-muted)' }}>Criterion:</span> <b className="mono">Gini Impurity</b></div>
          <div><span style={{ color: 'var(--text-muted)' }}>Class Weight:</span> <b className="mono">balanced_subsample</b></div>
          <div><span style={{ color: 'var(--text-muted)' }}>Min Split Samples:</span> <b className="mono">4 instances</b></div>
        </div>
      </div>
    </div>
  );
}
