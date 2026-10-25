import React from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { PlusCircle, Menu } from 'lucide-react';

const ROUTE_INFO = {
  '/': { title: 'Fleet Overview', subtitle: 'Live machine health telemetry & aggregate risk analytics' },
  '/predict': { title: 'Predict Equipment Failure', subtitle: 'Run continuous failure probability diagnostic estimation' },
  '/explorer': { title: 'Sensor Data Explorer', subtitle: 'Historical telemetry records, parameter distributions & export' },
  '/performance': { title: 'Model Performance', subtitle: 'Random Forest validation benchmarks, confusion matrix & ROC curves' },
  '/history': { title: 'Prediction History', subtitle: 'Session diagnostic records, failure probabilities & details' },
  '/about': { title: 'System Documentation & Help', subtitle: 'Engineering architecture, physical sensors & model methodology' }
};

export default function Topbar({ onToggleMobile }) {
  const location = useLocation();
  const navigate = useNavigate();

  const currentInfo = ROUTE_INFO[location.pathname] || {
    title: 'MachineGuard Telemetry',
    subtitle: 'Industrial Predictive Maintenance Intelligence'
  };

  return (
    <header className="topbar">
      <div className="topbar-left">
        <button
          className="mobile-menu-toggle"
          onClick={onToggleMobile}
          aria-label="Toggle mobile menu"
        >
          <Menu size={20} />
        </button>

        <div className="page-title-box">
          <h1 className="page-title-text">{currentInfo.title}</h1>
          <span className="page-subtitle-text">{currentInfo.subtitle}</span>
        </div>
      </div>

      <div className="topbar-right">
        {/* Simulation Mode Indicator Pill */}
        <div className="simulation-pill" title="Running in simulated offline telemetry demonstration mode">
          <span className="pulsing-cyan-dot" />
          <span className="pill-text-long">SIMULATION MODE • DEMO TELEMETRY</span>
        </div>

        {/* Quick CTA to Predict */}
        {location.pathname !== '/predict' && (
          <button
            className="btn-primary"
            onClick={() => navigate('/predict')}
            title="Launch new diagnostic prediction"
          >
            <PlusCircle size={15} />
            <span>New Prediction</span>
          </button>
        )}
      </div>
    </header>
  );
}
