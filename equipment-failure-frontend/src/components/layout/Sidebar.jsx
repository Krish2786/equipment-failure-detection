import React from 'react';
import { NavLink } from 'react-router-dom';
import {
  LayoutDashboard,
  Cpu,
  Database,
  BarChart3,
  History,
  HelpCircle,
  ChevronLeft,
  ChevronRight,
  ShieldAlert
} from 'lucide-react';

const NAV_ITEMS = [
  { path: '/', label: 'Overview', icon: LayoutDashboard },
  { path: '/predict', label: 'Predict Failure', icon: Cpu },
  { path: '/explorer', label: 'Sensor Data Explorer', icon: Database },
  { path: '/performance', label: 'Model Performance', icon: BarChart3 },
  { path: '/history', label: 'Prediction History', icon: History },
  { path: '/about', label: 'About / Help', icon: HelpCircle }
];

export default function Sidebar({ isCollapsed, onToggleCollapse, isMobileOpen, onCloseMobile }) {
  return (
    <aside
      className={`sidebar ${isCollapsed ? 'collapsed' : ''} ${isMobileOpen ? 'mobile-open' : ''}`}
      aria-label="Sidebar Navigation"
    >
      {/* Floating Border Toggle Button on the sidebar boundary */}
      <button
        type="button"
        className="sidebar-border-toggle"
        onClick={onToggleCollapse}
        title={isCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
        aria-label={isCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
      >
        {isCollapsed ? <ChevronRight size={14} /> : <ChevronLeft size={14} />}
      </button>

      {/* Sidebar Header */}
      <div className="sidebar-header">
        <div
          className="brand-wrapper"
          onClick={isCollapsed ? onToggleCollapse : undefined}
          style={isCollapsed ? { cursor: 'pointer' } : {}}
          title={isCollapsed ? 'Click to expand MachineGuard sidebar' : 'MachineGuard Industrial Suite'}
        >
          <div className="brand-icon-box">
            <ShieldAlert size={20} style={{ flexShrink: 0 }} />
          </div>
          {!isCollapsed && (
            <div className="brand-text">
              <span className="brand-title">MachineGuard</span>
              <span className="brand-subtitle">Telemetry AI</span>
            </div>
          )}
        </div>
      </div>

      {/* Navigation Items */}
      <nav className="sidebar-nav">
        {NAV_ITEMS.map((item) => {
          const Icon = item.icon;
          return (
            <NavLink
              key={item.path}
              to={item.path}
              className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}
              onClick={onCloseMobile}
              title={isCollapsed ? item.label : undefined}
            >
              <Icon className="nav-icon" style={{ flexShrink: 0 }} />
              {!isCollapsed && <span className="nav-label">{item.label}</span>}
            </NavLink>
          );
        })}
      </nav>

      {/* Sidebar Footer with System Status */}
      <div className="sidebar-footer">
        <div
          className="system-status-indicator"
          title="Random Forest probability estimator operational (RF-v1)"
        >
          <span className="status-dot-live" style={{ flexShrink: 0 }} />
          {!isCollapsed && (
            <div className="status-details">
              <span className="status-engine-name">RF-v1 Estimator</span>
              <span className="status-engine-state">STANDALONE DEMO</span>
            </div>
          )}
        </div>
      </div>
    </aside>
  );
}
