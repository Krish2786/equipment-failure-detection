import React from 'react';
import {
  BookOpen,
  Cpu,
  Layers,
  Thermometer,
  RotateCw,
  Gauge,
  Clock,
  AlertTriangle,
  CheckCircle,
  HelpCircle,
  TrendingUp,
  ShieldAlert
} from 'lucide-react';
import PageHeader from '../components/layout/PageHeader';

export default function AboutHelp() {
  return (
    <div className="fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '2rem', maxWidth: 1040, margin: '0 auto' }}>
      <PageHeader
        title="Engineering Architecture & System Documentation"
        description="Comprehensive technical reference for MachineGuard: Machine Learning System for Early Detection of Equipment Failure in Industrial Telemetry."
      />

      {/* Section 1: Executive Abstract & Problem Statement */}
      <section className="card">
        <div className="card-header">
          <div className="card-title-group">
            <h3 className="card-title">
              <BookOpen size={18} style={{ color: 'var(--accent-primary)' }} />
              <span>1. Executive Abstract & Industrial Problem Statement</span>
            </h3>
            <span className="card-subtitle">Addressing unscheduled downtime and catastrophic mechanical failure</span>
          </div>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem', fontSize: '0.9rem', lineHeight: 1.6, color: 'var(--text-secondary)' }}>
          <p>
            In modern computer-numerical-control (CNC) machining and automated industrial assembly, <b>unplanned machine downtime</b> is one of the highest operating expenses, often exceeding $10,000 to $250,000 per hour depending on plant throughput. Traditional maintenance regimens fluctuate between two suboptimal extremes:
          </p>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1rem', margin: '0.5rem 0' }}>
            <div style={{ padding: '1rem', background: 'var(--bg-elevated)', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-color)' }}>
              <b style={{ color: 'var(--risk-critical)', display: 'block', marginBottom: '0.35rem' }}>Reactive Maintenance (Run-to-Failure):</b>
              Components are used until catastrophic mechanical breakdown occurs. This ruins high-tolerance workpieces, damages secondary spindle assemblies, and poses severe operator safety hazards.
            </div>
            <div style={{ padding: '1rem', background: 'var(--bg-elevated)', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-color)' }}>
              <b style={{ color: 'var(--risk-moderate)', display: 'block', marginBottom: '0.35rem' }}>Calendar-Based Preventive Maintenance:</b>
              Components are replaced at rigid hourly intervals regardless of wear. This leads to premature discarding of expensive carbide tooling and unneeded labor overhead.
            </div>
          </div>

          <p>
            <b>MachineGuard</b> resolves this dichotomy by employing <b>machine learning on physical sensor telemetry</b> to detect microscopic degradation patterns before failure cascades into catastrophic seizure, enabling dynamic, condition-based maintenance scheduling.
          </p>
        </div>
      </section>

      {/* Section 2: The 5 Physical Sensor Inputs */}
      <section className="card">
        <div className="card-header">
          <div className="card-title-group">
            <h3 className="card-title">
              <Layers size={18} style={{ color: 'var(--accent-primary)' }} />
              <span>2. Physical Telemetry Sensors & Failure Mechanics</span>
            </h3>
            <span className="card-subtitle">How physical stress and degradation manifest in telemetry readouts</span>
          </div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '1rem' }}>
          {/* Air Temp */}
          <div style={{ padding: '1rem', background: 'var(--bg-elevated)', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-color)' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem', color: 'var(--accent-primary)' }}>
              <Thermometer size={16} />
              <b style={{ fontSize: '0.95rem' }}>Air Temperature [K] (290.0 – 310.0 K)</b>
            </div>
            <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
              Measures ambient shop-floor temperature. Governs the convective cooling boundary condition. Combined with Process Temperature, it forms the critical ΔT = T_process − T_air thermal gradient. When ΔT &lt; 8.6 K, convective heat removal is impaired, inducing <b>Heat Dissipation Failure (HDF)</b>.
            </p>
          </div>

          {/* Process Temp */}
          <div style={{ padding: '1rem', background: 'var(--bg-elevated)', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-color)' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem', color: 'var(--accent-primary)' }}>
              <Thermometer size={16} />
              <b style={{ fontSize: '0.95rem' }}>Process Temperature [K] (300.0 – 320.0 K)</b>
            </div>
            <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
              Internal cutting-zone temperature generated by friction between workpiece and tool. Sustained temperatures &gt; 313.0 K weaken cutting edge hardness, accelerate thermal expansion, and cause workpiece dimensional drift.
            </p>
          </div>

          {/* Rotational Speed */}
          <div style={{ padding: '1rem', background: 'var(--bg-elevated)', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-color)' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem', color: 'var(--risk-optimal)' }}>
              <RotateCw size={16} />
              <b style={{ fontSize: '0.95rem' }}>Rotational Speed [rpm] (1,000 – 3,000 rpm)</b>
            </div>
            <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
              Rotational velocity of the spindle motor. Extreme speeds (&gt; 2,600 rpm) create severe centrifugal radial loads on bearings. Conversely, low velocity under high cutting loads can trigger motor stall and stator winding overheating.
            </p>
          </div>

          {/* Torque */}
          <div style={{ padding: '1rem', background: 'var(--bg-elevated)', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-color)' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem', color: 'var(--risk-moderate)' }}>
              <Gauge size={16} />
              <b style={{ fontSize: '0.95rem' }}>Torque [Nm] (3.0 – 80.0 Nm)</b>
            </div>
            <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
              Torque exerted by the drive spindle. In normal operation, torque counterbalances cutting resistance. As cutting edges dull, required torque increases significantly. Values exceeding 60 Nm signify mechanical binding or hard inclusions.
            </p>
          </div>

          {/* Tool Wear */}
          <div style={{ padding: '1rem', background: 'var(--bg-elevated)', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-color)', gridColumn: 'span 1' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem', color: 'var(--risk-critical)' }}>
              <Clock size={16} />
              <b style={{ fontSize: '0.95rem' }}>Tool Wear [min] (0 – 260 min)</b>
            </div>
            <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
              Cumulative engaged cutting time for current tool insert. Flank wear accelerates past 180 min. Tool wear interacts multiplicatively with torque: when Torque × ToolWear &gt; 11,000 min·Nm, <b>Overstrain Failure (OSF)</b> occurs.
            </p>
          </div>
        </div>
      </section>

      {/* Section 3: Continuous Failure Probability Paradigm */}
      <section className="card">
        <div className="card-header">
          <div className="card-title-group">
            <h3 className="card-title">
              <TrendingUp size={18} style={{ color: 'var(--accent-primary)' }} />
              <span>3. Continuous Failure Probability (%) Paradigm vs Rigid Binary Output</span>
            </h3>
            <span className="card-subtitle">Why continuous probability estimation (predict_proba) transforms maintenance logistics</span>
          </div>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem', fontSize: '0.9rem', lineHeight: 1.6, color: 'var(--text-secondary)' }}>
          <p>
            Standard academic machine learning demonstrations often cast equipment diagnostics as a simplistic binary classification: <code>0 (No Failure)</code> or <code>1 (Failure)</code>. In real industrial environments, <b>binary outputs are operationally flawed</b>:
          </p>

          <ul style={{ paddingLeft: '1.25rem', display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
            <li>
              <b>Loss of Nuance:</b> A machine at 49% failure risk and a machine at 5% risk are both classified as "No Failure" by a standard 0.5 threshold, concealing rapid degradation.
            </li>
            <li>
              <b>Inability to Prioritize:</b> When a plant has 100 CNC machines, maintenance teams cannot fix all machines flagged as "Failure" simultaneously. They require an ordered probability ranking to dispatch resources where risk is highest.
            </li>
          </ul>

          <p>
            <b>MachineGuard</b> utilizes the continuous probability estimator (<code>predict_proba</code>) of the Random Forest model to calculate an exact continuous percentage ($0.0\%$ to $100.0\%$), and maps it into 4 actionable operational risk tiers:
          </p>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '0.75rem', marginTop: '0.5rem' }}>
            <div style={{ padding: '0.85rem', background: 'var(--risk-optimal-bg)', border: '1px solid var(--risk-optimal-border)', borderRadius: 'var(--radius-md)' }}>
              <b style={{ color: 'var(--risk-optimal)' }}>0.0% – 24.9%</b>
              <div style={{ fontWeight: 600, color: 'var(--text-primary)', margin: '0.2rem 0' }}>Optimal Operation</div>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Routine monitoring active; no maintenance required.</span>
            </div>

            <div style={{ padding: '0.85rem', background: 'var(--risk-moderate-bg)', border: '1px solid var(--risk-moderate-border)', borderRadius: 'var(--radius-md)' }}>
              <b style={{ color: 'var(--risk-moderate)' }}>25.0% – 49.9%</b>
              <div style={{ fontWeight: 600, color: 'var(--text-primary)', margin: '0.2rem 0' }}>Elevated Wear</div>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Schedule physical inspection in next routine shift window.</span>
            </div>

            <div style={{ padding: '0.85rem', background: 'var(--risk-high-bg)', border: '1px solid var(--risk-high-border)', borderRadius: 'var(--radius-md)' }}>
              <b style={{ color: 'var(--risk-high)' }}>50.0% – 74.9%</b>
              <div style={{ fontWeight: 600, color: 'var(--text-primary)', margin: '0.2rem 0' }}>High Failure Risk</div>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Prioritize preventive maintenance. Prepare replacement inserts.</span>
            </div>

            <div style={{ padding: '0.85rem', background: 'var(--risk-critical-bg)', border: '1px solid var(--risk-critical-border)', borderRadius: 'var(--radius-md)' }}>
              <b style={{ color: 'var(--risk-critical)' }}>75.0% – 100.0%</b>
              <div style={{ fontWeight: 600, color: 'var(--text-primary)', margin: '0.2rem 0' }}>Critical Imminent Risk</div>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Immediate intervention required. Halt cycle to prevent destruction.</span>
            </div>
          </div>
        </div>
      </section>

      {/* Section 4: Evaluation Metrics & Industrial Engineering Trade-Offs */}
      <section className="card">
        <div className="card-header">
          <div className="card-title-group">
            <h3 className="card-title">
              <ShieldAlert size={18} style={{ color: 'var(--accent-primary)' }} />
              <span>4. Interpreting Evaluation Metrics in Predictive Maintenance</span>
            </h3>
            <span className="card-subtitle">Why Recall and ROC-AUC are vastly more significant than Accuracy</span>
          </div>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem', fontSize: '0.9rem', lineHeight: 1.6, color: 'var(--text-secondary)' }}>
          <p>
            Industrial datasets are characterized by extreme <b>class imbalance</b> (typically 96.6% normal operation vs. 3.4% failure events). A trivial dummy classifier that always predicts "Normal" would achieve <b>96.6% Accuracy</b> while being completely useless for safety!
          </p>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
            <div style={{ padding: '0.75rem 1rem', background: 'var(--bg-elevated)', borderRadius: 'var(--radius-sm)', borderLeft: '3px solid var(--accent-primary)' }}>
              <b>Recall (Sensitivity = 92.1%):</b> The fraction of real breakdown events caught. In predictive maintenance, missing a real breakdown (False Negative) costs thousands of dollars in ruined spindles and emergency repairs.
            </div>
            <div style={{ padding: '0.75rem 1rem', background: 'var(--bg-elevated)', borderRadius: 'var(--radius-sm)', borderLeft: '3px solid var(--risk-optimal)' }}>
              <b>Precision (= 94.6%):</b> The percentage of failure alarms that are genuine. High precision ensures technicians do not develop "alarm fatigue" from constant false alerts.
            </div>
            <div style={{ padding: '0.75rem 1rem', background: 'var(--bg-elevated)', borderRadius: 'var(--radius-sm)', borderLeft: '3px solid var(--color-secondary)' }}>
              <b>ROC-AUC (= 96.4%):</b> Area under ROC curve confirms the model's ability to discriminate between degrading and healthy equipment across the entire continuum of probability thresholds.
            </div>
          </div>
        </div>
      </section>

      {/* Section 5: Operational Limitations */}
      <section className="card">
        <div className="card-header">
          <div className="card-title-group">
            <h3 className="card-title">
              <AlertTriangle size={18} style={{ color: 'var(--risk-moderate)' }} />
              <span>5. Operational Limitations & Deployment Considerations</span>
            </h3>
            <span className="card-subtitle">Real-world constraints to address prior to factory floor deployment</span>
          </div>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', fontSize: '0.88rem', lineHeight: 1.6, color: 'var(--text-secondary)' }}>
          <p>
            When transitioning MachineGuard from college engineering simulation to factory automation, the following engineering factors must be managed:
          </p>
          <ul style={{ paddingLeft: '1.25rem', display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
            <li><b>Sensor Calibration Drift:</b> RTD temperature probes and strain-gauge torque sensors drift over time due to thermal hysteresis, requiring periodic zero-point recalibration.</li>
            <li><b>Telemetry Jitter & Noise:</b> Electromagnetic interference (EMI) from high-power variable frequency drives (VFDs) introduces noise spikes into telemetry buses, necessitating Kalman filtering or moving-average smoothing.</li>
            <li><b>Workpiece Material Variance:</b> Cutting aerospace titanium (Ti-6Al-4V) exerts vastly different base torque loads than 6061-T6 aluminum; models must receive material hardness classification as a conditioning feature.</li>
          </ul>
        </div>
      </section>
    </div>
  );
}
