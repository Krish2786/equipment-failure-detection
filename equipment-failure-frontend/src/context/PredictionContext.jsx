import React, { createContext, useContext, useState, useEffect, useMemo } from 'react';
import { DEMO_PREDICTION_HISTORY } from '../data/demoPredictionHistory';
import { DEMO_SENSOR_RECORDS } from '../data/demoSensorData';

const STORAGE_KEY_PREDICTIONS = 'machineguard_predictions_v1';

export const PredictionContext = createContext(null);

export function PredictionProvider({ children }) {
  // Pure daylight enterprise theme
  const theme = 'light';

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', 'light');
  }, []);

  const toggleTheme = () => {
    // Theme locked to light mode per enterprise specification
  };

  // Prediction History: load from sessionStorage or fallback to preloaded DEMO_PREDICTION_HISTORY
  const [predictions, setPredictions] = useState(() => {
    try {
      const stored = sessionStorage.getItem(STORAGE_KEY_PREDICTIONS);
      if (stored) {
        const parsed = JSON.parse(stored);
        if (Array.isArray(parsed) && parsed.length > 0) {
          return parsed;
        }
      }
    } catch (e) {
      console.warn('Failed to parse predictions from sessionStorage', e);
    }
    return DEMO_PREDICTION_HISTORY;
  });

  // Sync predictions to sessionStorage
  useEffect(() => {
    try {
      sessionStorage.setItem(STORAGE_KEY_PREDICTIONS, JSON.stringify(predictions));
    } catch (e) {
      console.warn('Failed to persist predictions to sessionStorage', e);
    }
  }, [predictions]);

  // Add a newly computed prediction
  const addPrediction = (newPrediction) => {
    setPredictions(prev => [newPrediction, ...prev]);
  };

  // Clear or reset history
  const clearPredictions = () => {
    setPredictions([]);
    sessionStorage.removeItem(STORAGE_KEY_PREDICTIONS);
  };

  // Reset to default demo data
  const resetToDemoPredictions = () => {
    setPredictions(DEMO_PREDICTION_HISTORY);
    sessionStorage.setItem(STORAGE_KEY_PREDICTIONS, JSON.stringify(DEMO_PREDICTION_HISTORY));
  };

  // Computed KPIs across predictions
  const stats = useMemo(() => {
    const total = predictions.length;
    if (total === 0) {
      return {
        totalEvaluations: 0,
        meanFailureProbability: 0,
        highRiskAlertsCount: 0,
        riskDistribution: [
          { name: 'Optimal (<25%)', count: 0, percentage: 0, color: '#10B981' },
          { name: 'Elevated (25-49.9%)', count: 0, percentage: 0, color: '#F59E0B' },
          { name: 'High Risk (50-74.9%)', count: 0, percentage: 0, color: '#F97316' },
          { name: 'Critical (>=75%)', count: 0, percentage: 0, color: '#EF4444' }
        ],
        recentPredictions: []
      };
    }

    const sumProb = predictions.reduce((acc, p) => acc + (p.failurePercentage || (p.failureProbability * 100)), 0);
    const meanProb = Number((sumProb / total).toFixed(1));

    let optimalCount = 0;
    let moderateCount = 0;
    let highCount = 0;
    let criticalCount = 0;

    predictions.forEach(p => {
      const pct = p.failurePercentage || (p.failureProbability * 100);
      if (pct < 25) optimalCount++;
      else if (pct < 50) moderateCount++;
      else if (pct < 75) highCount++;
      else criticalCount++;
    });

    const highRiskAlertsCount = highCount + criticalCount;

    const riskDistribution = [
      {
        name: 'Optimal (<25%)',
        count: optimalCount,
        percentage: Number(((optimalCount / total) * 100).toFixed(1)),
        color: '#10B981'
      },
      {
        name: 'Elevated (25-49.9%)',
        count: moderateCount,
        percentage: Number(((moderateCount / total) * 100).toFixed(1)),
        color: '#F59E0B'
      },
      {
        name: 'High Risk (50-74.9%)',
        count: highCount,
        percentage: Number(((highCount / total) * 100).toFixed(1)),
        color: '#F97316'
      },
      {
        name: 'Critical (>=75%)',
        count: criticalCount,
        percentage: Number(((criticalCount / total) * 100).toFixed(1)),
        color: '#EF4444'
      }
    ];

    const recentPredictions = predictions.slice(0, 6);

    return {
      totalEvaluations: total,
      meanFailureProbability: meanProb,
      highRiskAlertsCount,
      riskDistribution,
      recentPredictions,
      modelRocAuc: 96.4
    };
  }, [predictions]);

  const value = {
    theme,
    toggleTheme,
    predictions,
    addPrediction,
    clearPredictions,
    resetToDemoPredictions,
    stats,
    allSensorRecords: DEMO_SENSOR_RECORDS
  };

  return (
    <PredictionContext.Provider value={value}>
      {children}
    </PredictionContext.Provider>
  );
}

export function usePredictionContext() {
  const context = useContext(PredictionContext);
  if (!context) {
    throw new Error('usePredictionContext must be used within a PredictionProvider');
  }
  return context;
}
