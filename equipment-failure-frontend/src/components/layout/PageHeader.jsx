import React from 'react';

export default function PageHeader({ title, description, actions, badge }) {
  return (
    <div className="page-header">
      <div className="page-header-title-area">
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', flexWrap: 'wrap' }}>
          <h2 className="page-header-title">{title}</h2>
          {badge}
        </div>
        {description && <p className="page-header-desc">{description}</p>}
      </div>
      {actions && <div className="page-header-actions">{actions}</div>}
    </div>
  );
}
