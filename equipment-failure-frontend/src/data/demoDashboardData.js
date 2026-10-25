/**
 * Dashboard telemetry trends and quick-fill presets.
 */

export const PRESET_CONFIGURATIONS = {
  nominal: {
    id: 'nominal',
    label: 'Nominal Baseline',
    description: 'Clean operating condition within optimal design envelope (<15% failure risk)',
    badge: 'Optimal',
    values: {
      airTemperature: 298.4,
      processTemperature: 308.7,
      rotationalSpeed: 1500,
      torque: 40.2,
      toolWear: 45
    }
  },
  thermalTorque: {
    id: 'thermalTorque',
    label: 'High Thermal & Torque Stress',
    description: 'Narrow delta-T dissipation gradient with elevated mechanical torque strain',
    badge: 'High Risk',
    values: {
      airTemperature: 302.5,
      processTemperature: 310.8,
      rotationalSpeed: 1350,
      torque: 64.5,
      toolWear: 175
    }
  },
  toolWearSevere: {
    id: 'toolWearSevere',
    label: 'Severe Tool Degradation',
    description: 'Extensive carbide blade fatigue (>220 min) combined with high overstrain factor',
    badge: 'Critical',
    values: {
      airTemperature: 298.5,
      processTemperature: 309.0,
      rotationalSpeed: 1320,
      torque: 68.0,
      toolWear: 242
    }
  }
};

/**
 * 24-Hour Telemetry Time Series (sampled every hour)
 * Used in SensorTrendChart.jsx
 */
export const TELEMETRY_TIME_SERIES = [
  { time: '00:00', airTemp: 297.8, processTemp: 307.9, rotationalSpeed: 1520, torque: 38.5, toolWear: 45, failureProbability: 4.8 },
  { time: '01:00', airTemp: 297.9, processTemp: 308.1, rotationalSpeed: 1515, torque: 39.0, toolWear: 53, failureProbability: 5.1 },
  { time: '02:00', airTemp: 298.0, processTemp: 308.2, rotationalSpeed: 1530, torque: 38.2, toolWear: 61, failureProbability: 5.0 },
  { time: '03:00', airTemp: 298.1, processTemp: 308.4, rotationalSpeed: 1500, torque: 40.1, toolWear: 69, failureProbability: 5.5 },
  { time: '04:00', airTemp: 298.0, processTemp: 308.3, rotationalSpeed: 1495, torque: 40.5, toolWear: 77, failureProbability: 5.7 },
  { time: '05:00', airTemp: 298.2, processTemp: 308.5, rotationalSpeed: 1510, torque: 39.8, toolWear: 85, failureProbability: 6.2 },
  { time: '06:00', airTemp: 298.4, processTemp: 308.8, rotationalSpeed: 1485, torque: 41.6, toolWear: 93, failureProbability: 7.0 },
  { time: '07:00', airTemp: 298.7, processTemp: 309.1, rotationalSpeed: 1470, torque: 43.2, toolWear: 101, failureProbability: 8.4 },
  { time: '08:00', airTemp: 299.1, processTemp: 309.5, rotationalSpeed: 1455, torque: 45.0, toolWear: 110, failureProbability: 10.2 },
  { time: '09:00', airTemp: 299.5, processTemp: 310.0, rotationalSpeed: 1440, torque: 47.4, toolWear: 119, failureProbability: 13.5 },
  { time: '10:00', airTemp: 300.0, processTemp: 310.6, rotationalSpeed: 1425, torque: 50.1, toolWear: 128, failureProbability: 17.8 },
  { time: '11:00', airTemp: 300.4, processTemp: 311.2, rotationalSpeed: 1410, torque: 52.8, toolWear: 137, failureProbability: 22.4 },
  { time: '12:00', airTemp: 300.9, processTemp: 311.8, rotationalSpeed: 1395, torque: 55.4, toolWear: 146, failureProbability: 28.6 },
  { time: '13:00', airTemp: 301.2, processTemp: 312.2, rotationalSpeed: 1380, torque: 57.9, toolWear: 155, failureProbability: 35.1 },
  { time: '14:00', airTemp: 301.5, processTemp: 312.6, rotationalSpeed: 1365, torque: 60.5, toolWear: 165, failureProbability: 44.8 },
  { time: '15:00', airTemp: 301.8, processTemp: 313.1, rotationalSpeed: 1350, torque: 63.2, toolWear: 176, failureProbability: 56.4 },
  { time: '16:00', airTemp: 302.0, processTemp: 313.5, rotationalSpeed: 1335, torque: 66.0, toolWear: 188, failureProbability: 68.2 },
  { time: '17:00', airTemp: 301.7, processTemp: 313.0, rotationalSpeed: 1345, torque: 64.8, toolWear: 198, failureProbability: 63.5 },
  { time: '18:00', airTemp: 301.2, processTemp: 312.4, rotationalSpeed: 1360, torque: 61.5, toolWear: 206, failureProbability: 54.0 },
  { time: '19:00', airTemp: 300.6, processTemp: 311.6, rotationalSpeed: 1385, torque: 56.2, toolWear: 214, failureProbability: 46.2 },
  { time: '20:00', airTemp: 300.0, processTemp: 310.8, rotationalSpeed: 1415, torque: 51.0, toolWear: 222, failureProbability: 39.5 },
  { time: '21:00', airTemp: 299.4, processTemp: 310.0, rotationalSpeed: 1445, torque: 46.5, toolWear: 230, failureProbability: 32.1 },
  { time: '22:00', airTemp: 298.8, processTemp: 309.2, rotationalSpeed: 1475, torque: 42.8, toolWear: 236, failureProbability: 25.8 },
  { time: '23:00', airTemp: 298.2, processTemp: 308.6, rotationalSpeed: 1500, torque: 40.0, toolWear: 242, failureProbability: 21.0 }
];
