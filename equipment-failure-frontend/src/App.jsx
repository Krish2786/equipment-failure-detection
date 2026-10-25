import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { PredictionProvider } from './context/PredictionContext';
import MainLayout from './components/layout/MainLayout';

import Overview from './pages/Overview';
import PredictFailure from './pages/PredictFailure';
import SensorDataExplorer from './pages/SensorDataExplorer';
import ModelPerformance from './pages/ModelPerformance';
import PredictionHistory from './pages/PredictionHistory';
import AboutHelp from './pages/AboutHelp';

export default function App() {
  return (
    <PredictionProvider>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<MainLayout />}>
            <Route index element={<Overview />} />
            <Route path="overview" element={<Navigate to="/" replace />} />
            <Route path="predict" element={<PredictFailure />} />
            <Route path="explorer" element={<SensorDataExplorer />} />
            <Route path="performance" element={<ModelPerformance />} />
            <Route path="history" element={<PredictionHistory />} />
            <Route path="about" element={<AboutHelp />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </PredictionProvider>
  );
}
