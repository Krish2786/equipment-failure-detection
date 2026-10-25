import React, { useState } from 'react';
import { Outlet } from 'react-router-dom';
import Sidebar from './Sidebar';
import Topbar from './Topbar';

export default function MainLayout() {
  const [isCollapsed, setIsCollapsed] = useState(false);
  const [isMobileOpen, setIsMobileOpen] = useState(false);

  return (
    <div className="app-layout">
      {/* Sidebar Navigation */}
      <Sidebar
        isCollapsed={isCollapsed}
        onToggleCollapse={() => setIsCollapsed(!isCollapsed)}
        isMobileOpen={isMobileOpen}
        onCloseMobile={() => setIsMobileOpen(false)}
      />

      {/* Main Content Area */}
      <div className={`main-wrapper ${isCollapsed ? 'expanded' : ''}`}>
        <Topbar onToggleMobile={() => setIsMobileOpen(!isMobileOpen)} />
        <main className="page-container">
          <Outlet />
        </main>
      </div>

      {/* Mobile Backdrop overlay */}
      {isMobileOpen && (
        <div
          className="modal-overlay"
          style={{ zIndex: 95 }}
          onClick={() => setIsMobileOpen(false)}
        />
      )}
    </div>
  );
}
