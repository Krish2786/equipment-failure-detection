import React, { useState } from 'react';
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  TrendingUp,
  Cpu,
  Info,
  X,
  Gauge
} from 'lucide-react';
import { usePredictionContext } from '../context/PredictionContext';
import { formatPercentage } from '../utils/formatters';
import StatCard from '../components/dashboard/StatCard';
import EquipmentStatusCard from '../components/dashboard/EquipmentStatusCard';
import SensorTrendChart from '../components/dashboard/SensorTrendChart';
import RiskDistributionChart from '../components/dashboard/RiskDistributionChart';
import RecentPredictionsTable from '../components/dashboard/RecentPredictionsTable';

export default function Overview() {
  const { stats, allSensorRecords } = usePredictionContext();
  const [showBanner, setShowBanner] = useState(true);

  return (
    <div className="fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      {/* Dismissible Simulation Disclaimer Banner */}
      {showBanner && (
        <div className="disclaimer-banner">
          <div className="disclaimer-content">
            <Info size={18} className="disclaimer-icon" />
            <span>
              <strong>Demonstration Environment:</strong> Telemetry streams and diagnostic metrics represent simulated datasets modeled after the AI4I 2020 Predictive Maintenance standard, pending live Scikit-learn FastAPI backend connection.
            </span>
          </div>
          <button
            className="btn-outline btn-sm btn-icon-only"
            onClick={() => setShowBanner(false)}
            aria-label="Dismiss banner"
            style={{ border: 'none', background: 'transparent' }}
          >
            <X size={16} />
          </button>
        </div>
      )}

      {/* 4 KPI Stat Cards */}
      <div className="grid-4">
        <StatCard
          label="Total Evaluations"
          value={stats.totalEvaluations}
          icon={Activity}
          trend="+100% Session"
          trendDirection="positive"
          subtext="Telemetry runs logged"
        />

        <StatCard
          label="Mean Failure Probability"
          value={formatPercentage(stats.meanFailureProbability)}
          icon={Gauge}
          trend={stats.meanFailureProbability > 50 ? 'High' : 'Moderate'}
          trendDirection={stats.meanFailureProbability > 50 ? 'negative' : 'positive'}
          subtext="Fleet average failure risk"
        />

        <StatCard
          label="High/Critical Risk Alerts"
          value={stats.highRiskAlertsCount}
          icon={AlertTriangle}
          trend={`${stats.highRiskAlertsCount} Units`}
          trendDirection={stats.highRiskAlertsCount > 3 ? 'negative' : 'neutral'}
          subtext=">= 50% failure probability"
        />

        <StatCard
          label="Model ROC-AUC Score"
          value={`${stats.modelRocAuc}%`}
          icon={TrendingUp}
          trend="RF Benchmark"
          trendDirection="positive"
          subtext="Discrimination accuracy"
        />
      </div>

      {/* Fleet Operational Health Summary Card */}
      <EquipmentStatusCard records={allSensorRecords} />

      {/* Dual Analytics Row: 24h Trend Chart (2/3 width) and Risk Distribution (1/3 width) */}
      <div className="grid-2-1">
        <SensorTrendChart />
        <RiskDistributionChart />
      </div>

      {/* Recent Evaluations Table */}
      <RecentPredictionsTable />
    </div>
  );
}
