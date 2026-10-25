import React from 'react';
import { Cpu, Sparkles, AlertCircle } from 'lucide-react';
import { usePrediction } from '../hooks/usePrediction';
import PageHeader from '../components/layout/PageHeader';
import SensorInputForm from '../components/prediction/SensorInputForm';
import PredictionResultGauge from '../components/prediction/PredictionResultGauge';

export default function PredictFailure() {
  const {
    sensorValues,
    errors,
    activePreset,
    isEvaluating,
    lastResult,
    evaluationError,
    handleSensorChange,
    applyPreset,
    resetForm,
    runPrediction
  } = usePrediction();

  return (
    <div className="fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      <PageHeader
        title="Predict Equipment Failure"
        description="Compute real-time failure probability percentages using continuous Scikit-learn Random Forest regression heuristics based on 5 physical telemetry variables."
        badge={
          <span className="risk-badge tier-optimal">
            <span className="risk-badge-dot" />
            <span>Random Forest • continuous predict_proba</span>
          </span>
        }
      />

      {/* Two-Column Split Layout */}
      <div className="grid-split-equal">
        {/* Left Column: Sensor Input Form */}
        <div>
          <SensorInputForm
            sensorValues={sensorValues}
            onChange={handleSensorChange}
            onApplyPreset={applyPreset}
            activePreset={activePreset}
            onReset={resetForm}
            onSubmit={runPrediction}
            isEvaluating={isEvaluating}
            errors={errors}
          />
        </div>

        {/* Right Column: Diagnostic Output Panel */}
        <div>
          <PredictionResultGauge
            result={lastResult}
            isEvaluating={isEvaluating}
            error={evaluationError}
          />
        </div>
      </div>
    </div>
  );
}
