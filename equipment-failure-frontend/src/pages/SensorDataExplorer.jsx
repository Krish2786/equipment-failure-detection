import React, { useState, useMemo } from 'react';
import {
  Download,
  Search,
  Filter,
  SlidersHorizontal,
  ChevronLeft,
  ChevronRight,
  Database,
  BarChart2
} from 'lucide-react';
import {
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  ZAxis,
  Tooltip,
  CartesianGrid,
  BarChart,
  Bar,
  Cell
} from 'recharts';
import { DEMO_SENSOR_RECORDS } from '../data/demoSensorData';
import { formatPercentage, formatTimestamp } from '../utils/formatters';
import PageHeader from '../components/layout/PageHeader';
import RiskBadge from '../components/common/RiskBadge';
import EmptyState from '../components/common/EmptyState';

export default function SensorDataExplorer() {
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedTier, setSelectedTier] = useState('ALL');
  const [maxProbFilter, setMaxProbFilter] = useState(100);
  const [currentPage, setCurrentPage] = useState(1);
  const pageSize = 10;

  // Filter records based on controls
  const filteredRecords = useMemo(() => {
    return DEMO_SENSOR_RECORDS.filter((rec) => {
      // Search term
      const matchesSearch =
        rec.sampleId.toLowerCase().includes(searchTerm.toLowerCase()) ||
        rec.equipmentName.toLowerCase().includes(searchTerm.toLowerCase()) ||
        rec.equipmentType.toLowerCase().includes(searchTerm.toLowerCase());

      // Risk tier
      const matchesTier =
        selectedTier === 'ALL' ||
        rec.riskTier.toLowerCase().includes(selectedTier.toLowerCase());

      // Max probability
      const matchesProb = rec.failureProbability <= maxProbFilter;

      return matchesSearch && matchesTier && matchesProb;
    });
  }, [searchTerm, selectedTier, maxProbFilter]);

  // Pagination slice
  const totalPages = Math.ceil(filteredRecords.length / pageSize) || 1;
  const paginatedRecords = useMemo(() => {
    const start = (currentPage - 1) * pageSize;
    return filteredRecords.slice(start, start + pageSize);
  }, [filteredRecords, currentPage, pageSize]);

  // Comparative Scatter Data: Torque vs Rotational Speed
  const scatterData = useMemo(() => {
    return filteredRecords.map((r) => ({
      x: r.rotationalSpeed,
      y: r.torque,
      name: r.sampleId,
      prob: r.failureProbability,
      color: r.failureProbability >= 75 ? '#EF4444' : r.failureProbability >= 50 ? '#F97316' : r.failureProbability >= 25 ? '#F59E0B' : '#10B981'
    }));
  }, [filteredRecords]);

  // Comparative Tool Wear by Risk Tier aggregation
  const toolWearDistribution = useMemo(() => {
    const groups = {
      'Optimal (<25%)': { totalWear: 0, count: 0, color: '#10B981' },
      'Elevated (25-49.9%)': { totalWear: 0, count: 0, color: '#F59E0B' },
      'High Risk (50-74.9%)': { totalWear: 0, count: 0, color: '#F97316' },
      'Critical (>=75%)': { totalWear: 0, count: 0, color: '#EF4444' }
    };

    filteredRecords.forEach((r) => {
      if (r.failureProbability < 25) {
        groups['Optimal (<25%)'].totalWear += r.toolWear;
        groups['Optimal (<25%)'].count += 1;
      } else if (r.failureProbability < 50) {
        groups['Elevated (25-49.9%)'].totalWear += r.toolWear;
        groups['Elevated (25-49.9%)'].count += 1;
      } else if (r.failureProbability < 75) {
        groups['High Risk (50-74.9%)'].totalWear += r.toolWear;
        groups['High Risk (50-74.9%)'].count += 1;
      } else {
        groups['Critical (>=75%)'].totalWear += r.toolWear;
        groups['Critical (>=75%)'].count += 1;
      }
    });

    return Object.keys(groups).map((key) => ({
      tier: key,
      avgWear: groups[key].count ? Math.round(groups[key].totalWear / groups[key].count) : 0,
      color: groups[key].color
    }));
  }, [filteredRecords]);

  // CSV Export Function
  const handleExportCSV = () => {
    const headers = [
      'Sample ID',
      'Equipment Name',
      'Equipment Type',
      'Air Temperature (K)',
      'Process Temperature (K)',
      'Rotational Speed (rpm)',
      'Torque (Nm)',
      'Tool Wear (min)',
      'Failure Probability (%)',
      'Risk Tier',
      'Status'
    ];

    const rows = filteredRecords.map((r) => [
      `"${r.sampleId}"`,
      `"${r.equipmentName}"`,
      `"${r.equipmentType}"`,
      r.airTemperature,
      r.processTemperature,
      r.rotationalSpeed,
      r.torque,
      r.toolWear,
      r.failureProbability,
      `"${r.riskTier}"`,
      `"${r.status}"`
    ]);

    const csvContent = 'data:text/csv;charset=utf-8,' + [headers.join(','), ...rows.map(e => e.join(','))].join('\n');
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', `machineguard_telemetry_export_${new Date().toISOString().slice(0, 10)}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      <PageHeader
        title="Sensor Data Explorer"
        description="Filter, inspect, and benchmark 40 continuous industrial telemetry records. Discover operating envelope clusters and export datasets for offline modeling."
        actions={
          <button className="btn-primary btn-sm" onClick={handleExportCSV}>
            <Download size={15} />
            <span>Export Filtered CSV ({filteredRecords.length})</span>
          </button>
        }
      />

      {/* Dataset Filter & Control Toolbar */}
      <div className="card" style={{ padding: '1.25rem' }}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '1rem', alignItems: 'flex-end' }}>
          {/* Search Input */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.35rem' }}>
            <label style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)' }}>
              Search Telemetry / Sample ID:
            </label>
            <div style={{ position: 'relative' }}>
              <Search size={16} style={{ position: 'absolute', left: 10, top: '50%', transform: 'translateY(-50%)', color: 'var(--text-muted)' }} />
              <input
                type="text"
                placeholder="Search EQ-1001, Lathe, Mill..."
                value={searchTerm}
                onChange={(e) => { setSearchTerm(e.target.value); setCurrentPage(1); }}
                style={{ width: '100%', paddingLeft: '2.2rem' }}
              />
            </div>
          </div>

          {/* Risk Tier Filter */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.35rem' }}>
            <label style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)' }}>
              Filter by Operational Risk Tier:
            </label>
            <select
              value={selectedTier}
              onChange={(e) => { setSelectedTier(e.target.value); setCurrentPage(1); }}
            >
              <option value="ALL">All Risk Tiers</option>
              <option value="Optimal">Optimal Operation (&lt;25%)</option>
              <option value="Elevated">Elevated Wear (25-49.9%)</option>
              <option value="High">High Failure Risk (50-74.9%)</option>
              <option value="Critical">Critical Imminent Risk (&gt;=75%)</option>
            </select>
          </div>

          {/* Probability Upper Bound Filter */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.35rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
              <span style={{ fontWeight: 600 }}>Max Failure Probability:</span>
              <span className="mono" style={{ color: 'var(--accent-primary)', fontWeight: 600 }}>&le; {maxProbFilter}%</span>
            </div>
            <input
              type="range"
              min="10"
              max="100"
              step="5"
              value={maxProbFilter}
              onChange={(e) => { setMaxProbFilter(Number(e.target.value)); setCurrentPage(1); }}
              className="sensor-range-slider"
            />
          </div>
        </div>
      </div>

      {/* Dual Comparative Analytics Visualizations */}
      <div className="grid-2">
        {/* Visual 1: Torque vs Speed Envelope Scatter */}
        <div className="card">
          <div className="card-header">
            <div className="card-title-group">
              <h3 className="card-title">
                <BarChart2 size={16} style={{ color: 'var(--accent-primary)' }} />
                <span>Torque vs. Rotational Speed Envelope</span>
              </h3>
              <span className="card-subtitle">Telemetry clustering colored by continuous risk tier</span>
            </div>
          </div>
          <div style={{ height: 240, width: '100%' }}>
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart margin={{ top: 10, right: 10, bottom: 10, left: -10 }}>
                <CartesianGrid stroke="#E2E8F0" strokeDasharray="3 3" />
                <XAxis
                  type="number"
                  dataKey="x"
                  name="Speed"
                  unit=" rpm"
                  domain={[1250, 1650]}
                  tick={{ fill: '#64748B', fontSize: 11, fontFamily: 'var(--font-mono)' }}
                  axisLine={{ stroke: '#E2E8F0' }}
                  tickLine={false}
                />
                <YAxis
                  type="number"
                  dataKey="y"
                  name="Torque"
                  unit=" Nm"
                  domain={[25, 75]}
                  tick={{ fill: '#64748B', fontSize: 11, fontFamily: 'var(--font-mono)' }}
                  axisLine={false}
                  tickLine={false}
                />
                <Tooltip
                  cursor={{ strokeDasharray: '3 3' }}
                  content={({ active, payload }) => {
                    if (active && payload && payload.length) {
                      const data = payload[0].payload;
                      return (
                        <div className="custom-recharts-tooltip">
                          <div className="tooltip-header">{data.name}</div>
                          <div className="tooltip-row">
                            <span>Speed:</span>
                            <b className="mono">{data.x} rpm</b>
                          </div>
                          <div className="tooltip-row">
                            <span>Torque:</span>
                            <b className="mono">{data.y} Nm</b>
                          </div>
                          <div className="tooltip-row">
                            <span>Risk Prob.:</span>
                            <b className="mono" style={{ color: data.color }}>{data.prob}%</b>
                          </div>
                        </div>
                      );
                    }
                    return null;
                  }}
                />
                <Scatter name="Telemetry" data={scatterData}>
                  {scatterData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Scatter>
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Visual 2: Tool Wear vs Risk Tier */}
        <div className="card">
          <div className="card-header">
            <div className="card-title-group">
              <h3 className="card-title">
                <BarChart2 size={16} style={{ color: 'var(--accent-primary)' }} />
                <span>Mean Tool Wear (min) across Risk Tiers</span>
              </h3>
              <span className="card-subtitle">Correlation between carbide blade wear and failure likelihood</span>
            </div>
          </div>
          <div style={{ height: 240, width: '100%' }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={toolWearDistribution} margin={{ top: 10, right: 10, bottom: 10, left: -10 }}>
                <CartesianGrid stroke="#E2E8F0" strokeDasharray="3 3" vertical={false} />
                <XAxis
                  dataKey="tier"
                  tick={{ fill: '#64748B', fontSize: 11 }}
                  axisLine={{ stroke: '#E2E8F0' }}
                  tickLine={false}
                />
                <YAxis
                  domain={[0, 250]}
                  unit=" min"
                  tick={{ fill: '#64748B', fontSize: 11, fontFamily: 'var(--font-mono)' }}
                  axisLine={false}
                  tickLine={false}
                />
                <Tooltip
                  content={({ active, payload }) => {
                    if (active && payload && payload.length) {
                      const data = payload[0].payload;
                      return (
                        <div className="custom-recharts-tooltip">
                          <div className="tooltip-header">{data.tier}</div>
                          <div className="tooltip-row">
                            <span>Average Tool Wear:</span>
                            <b className="mono">{data.avgWear} min</b>
                          </div>
                        </div>
                      );
                    }
                    return null;
                  }}
                />
                <Bar dataKey="avgWear" radius={[4, 4, 0, 0]}>
                  {toolWearDistribution.map((entry, index) => (
                    <Cell key={`bar-${index}`} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Interactive Telemetry Table */}
      <div className="card">
        <div className="card-header">
          <div className="card-title-group">
            <h3 className="card-title">
              <Database size={16} style={{ color: 'var(--color-primary)' }} />
              <span>Telemetry Master Stream</span>
            </h3>
            <span className="card-subtitle">
              Showing {filteredRecords.length > 0 ? (currentPage - 1) * pageSize + 1 : 0} – {Math.min(currentPage * pageSize, filteredRecords.length)} of {filteredRecords.length} filtered records
            </span>
          </div>
        </div>

        {filteredRecords.length === 0 ? (
          <EmptyState
            title="No Matching Telemetry Records"
            description="Adjust your search query or reset the risk tier / probability filters to view records."
            actionLabel="Reset Filters"
            onAction={() => { setSearchTerm(''); setSelectedTier('ALL'); setMaxProbFilter(100); }}
          />
        ) : (
          <>
            <div className="table-container">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Sample ID</th>
                    <th>Equipment Type</th>
                    <th>Air Temp</th>
                    <th>Process Temp</th>
                    <th>Speed</th>
                    <th>Torque</th>
                    <th>Tool Wear</th>
                    <th>Failure Prob.</th>
                    <th>Risk Tier</th>
                    <th>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {paginatedRecords.map((r) => (
                    <tr key={r.id}>
                      <td>
                        <span className="mono" style={{ fontWeight: 600, color: 'var(--color-primary)' }}>
                          {r.sampleId}
                        </span>
                      </td>
                      <td>
                        <div style={{ display: 'flex', flexDirection: 'column' }}>
                          <span style={{ fontWeight: 500, color: 'var(--text-primary)' }}>{r.equipmentName}</span>
                          <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>{r.equipmentType}</span>
                        </div>
                      </td>
                      <td><span className="mono">{r.airTemperature} K</span></td>
                      <td><span className="mono">{r.processTemperature} K</span></td>
                      <td><span className="mono">{r.rotationalSpeed} rpm</span></td>
                      <td><span className="mono">{r.torque} Nm</span></td>
                      <td><span className="mono">{r.toolWear} min</span></td>
                      <td>
                        <span className="mono" style={{ fontWeight: 700, color: r.failureProbability >= 50 ? 'var(--risk-critical)' : 'var(--text-primary)' }}>
                          {formatPercentage(r.failureProbability)}
                        </span>
                      </td>
                      <td>
                        <RiskBadge probability={r.failureProbability} />
                      </td>
                      <td>
                        <span className="mono" style={{ fontSize: '0.75rem', color: r.status === 'Active' ? 'var(--risk-optimal)' : r.status === 'Warning' ? 'var(--risk-high)' : 'var(--risk-critical)' }}>
                          {r.status}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Pagination Toolbar */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '1.25rem', paddingTop: '0.75rem', borderTop: '1px solid var(--border-color)', flexWrap: 'wrap', gap: '0.75rem' }}>
              <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                Page <b className="mono">{currentPage}</b> of <b className="mono">{totalPages}</b>
              </span>

              <div style={{ display: 'flex', gap: '0.5rem' }}>
                <button
                  className="btn-secondary btn-sm"
                  onClick={() => setCurrentPage(p => Math.max(p - 1, 1))}
                  disabled={currentPage === 1}
                >
                  <ChevronLeft size={15} />
                  <span>Previous</span>
                </button>
                <button
                  className="btn-secondary btn-sm"
                  onClick={() => setCurrentPage(p => Math.min(p + 1, totalPages))}
                  disabled={currentPage === totalPages}
                >
                  <span>Next</span>
                  <ChevronRight size={15} />
                </button>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
