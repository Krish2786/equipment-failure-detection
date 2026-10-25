/**
 * Preloaded Historical Prediction Records.
 * Demonstrates past diagnostic runs across various equipment types.
 */

export const DEMO_PREDICTION_HISTORY = [
  {
    id: 'PRED-1049',
    sampleId: 'SAMPLE-9421',
    equipmentName: 'CNC Milling Alpha',
    failureProbability: 0.884,
    failurePercentage: 88.4,
    riskTier: 'Critical Imminent Risk',
    riskLevel: 'critical',
    timestamp: '2026-10-25T14:12:00Z',
    modelVersion: 'rf-sklearn-demo-v1',
    isDemo: true,
    contributingFactors: [
      'Critical Torque-ToolWear overstrain index (15,458 > 11,000 threshold)',
      'Severe cutting tool wear degradation (228 min > 210 min replacement threshold)',
      'Extreme mechanical torque load (67.8 Nm exceeding high-strain limit of 62.0 Nm)'
    ],
    recommendedAction: 'Immediate intervention recommended. Halt cycle or dispatch maintenance engineer immediately.',
    sensors: {
      airTemperature: 298.4,
      processTemperature: 308.9,
      rotationalSpeed: 1332,
      torque: 67.8,
      toolWear: 228,
      deltaT: 10.5,
      powerWatts: 9456
    }
  },
  {
    id: 'PRED-1048',
    sampleId: 'SAMPLE-8172',
    equipmentName: 'Hydraulic Stamping Press 1',
    failureProbability: 0.124,
    failurePercentage: 12.4,
    riskTier: 'Optimal Operation',
    riskLevel: 'optimal',
    timestamp: '2026-10-25T13:45:10Z',
    modelVersion: 'rf-sklearn-demo-v1',
    isDemo: true,
    contributingFactors: [
      'All sensor telemetry parameters operate well within nominal baseline envelopes',
      'Thermal dissipation rate, torque strain, and tool wear indices are within safety limits'
    ],
    recommendedAction: 'Routine monitoring active. No intervention required.',
    sensors: {
      airTemperature: 298.2,
      processTemperature: 308.7,
      rotationalSpeed: 1510,
      torque: 41.5,
      toolWear: 64,
      deltaT: 10.5,
      powerWatts: 6562
    }
  },
  {
    id: 'PRED-1047',
    sampleId: 'SAMPLE-7320',
    equipmentName: '5-Axis CNC Mill Gamma',
    failureProbability: 0.768,
    failurePercentage: 76.8,
    riskTier: 'Critical Imminent Risk',
    riskLevel: 'critical',
    timestamp: '2026-10-25T13:10:25Z',
    modelVersion: 'rf-sklearn-demo-v1',
    isDemo: true,
    contributingFactors: [
      'Critical Torque-ToolWear overstrain index (12,691 > 11,000 threshold)',
      'Extreme mechanical torque load (64.1 Nm exceeding high-strain limit of 62.0 Nm)',
      'Elevated tool wear (198 min); tool nearing end of optimal blade sharpness window'
    ],
    recommendedAction: 'Immediate intervention recommended. Halt cycle or dispatch maintenance engineer immediately.',
    sensors: {
      airTemperature: 298.7,
      processTemperature: 309.4,
      rotationalSpeed: 1390,
      torque: 64.1,
      toolWear: 198,
      deltaT: 10.7,
      powerWatts: 9330
    }
  },
  {
    id: 'PRED-1046',
    sampleId: 'SAMPLE-6291',
    equipmentName: 'HMC Lathe Unit 05',
    failureProbability: 0.513,
    failurePercentage: 51.3,
    riskTier: 'High Failure Risk',
    riskLevel: 'high',
    timestamp: '2026-10-25T12:20:00Z',
    modelVersion: 'rf-sklearn-demo-v1',
    isDemo: true,
    contributingFactors: [
      'Elevated tool wear (185 min); tool nearing end of optimal blade sharpness window',
      'Elevated process thermal gradient: Delta-T is high (10.8 K)',
      'Mechanical power factor approaching duty limit'
    ],
    recommendedAction: 'Prioritize preventive maintenance. Inspect cutting tool, bearings, and thermal loops.',
    sensors: {
      airTemperature: 299.4,
      processTemperature: 310.2,
      rotationalSpeed: 1380,
      torque: 56.7,
      toolWear: 185,
      deltaT: 10.8,
      powerWatts: 8193
    }
  },
  {
    id: 'PRED-1045',
    sampleId: 'SAMPLE-5514',
    equipmentName: 'Precision Lathe P-01',
    failureProbability: 0.048,
    failurePercentage: 4.8,
    riskTier: 'Optimal Operation',
    riskLevel: 'optimal',
    timestamp: '2026-10-25T11:35:40Z',
    modelVersion: 'rf-sklearn-demo-v1',
    isDemo: true,
    contributingFactors: [
      'All sensor telemetry parameters operate well within nominal baseline envelopes'
    ],
    recommendedAction: 'Routine monitoring active. No intervention required.',
    sensors: {
      airTemperature: 298.8,
      processTemperature: 309.5,
      rotationalSpeed: 1580,
      torque: 34.2,
      toolWear: 50,
      deltaT: 10.7,
      powerWatts: 5658
    }
  },
  {
    id: 'PRED-1044',
    sampleId: 'SAMPLE-4890',
    equipmentName: 'Hydraulic Stamping Press 2',
    failureProbability: 0.465,
    failurePercentage: 46.5,
    riskTier: 'Elevated Wear',
    riskLevel: 'moderate',
    timestamp: '2026-10-25T10:50:15Z',
    modelVersion: 'rf-sklearn-demo-v1',
    isDemo: true,
    contributingFactors: [
      'Mechanical strain detected: High torque of 58.2 Nm',
      'Tool wear progressing: 172 min approaching critical window'
    ],
    recommendedAction: 'Schedule physical inspection during the next planned maintenance window.',
    sensors: {
      airTemperature: 298.6,
      processTemperature: 309.2,
      rotationalSpeed: 1445,
      torque: 58.2,
      toolWear: 172,
      deltaT: 10.6,
      powerWatts: 8806
    }
  },
  {
    id: 'PRED-1043',
    sampleId: 'SAMPLE-3912',
    equipmentName: '5-Axis CNC Mill Delta',
    failureProbability: 0.615,
    failurePercentage: 61.5,
    riskTier: 'High Failure Risk',
    riskLevel: 'high',
    timestamp: '2026-10-25T09:40:00Z',
    modelVersion: 'rf-sklearn-demo-v1',
    isDemo: true,
    contributingFactors: [
      'Heat dissipation deficiency: Delta-T is low (8.1 K < 8.6 K nominal limit), causing internal heat accumulation',
      'Elevated torque load of 52.4 Nm'
    ],
    recommendedAction: 'Prioritize preventive maintenance. Inspect cutting tool, bearings, and thermal loops.',
    sensors: {
      airTemperature: 300.1,
      processTemperature: 308.2,
      rotationalSpeed: 1420,
      torque: 52.4,
      toolWear: 130,
      deltaT: 8.1,
      powerWatts: 7791
    }
  },
  {
    id: 'PRED-1042',
    sampleId: 'SAMPLE-2804',
    equipmentName: 'Industrial Spindle Rig 2',
    failureProbability: 0.841,
    failurePercentage: 84.1,
    riskTier: 'Critical Imminent Risk',
    riskLevel: 'critical',
    timestamp: '2026-10-25T08:15:30Z',
    modelVersion: 'rf-sklearn-demo-v1',
    isDemo: true,
    contributingFactors: [
      'Critical Torque-ToolWear overstrain index (14,227 > 11,000 threshold)',
      'Extreme mechanical torque load (69.4 Nm exceeding high-strain limit of 62.0 Nm)',
      'Excessive mechanical spindle power (9,737 W > 8,500 W duty rating)'
    ],
    recommendedAction: 'Immediate intervention recommended. Halt cycle or dispatch maintenance engineer immediately.',
    sensors: {
      airTemperature: 299.5,
      processTemperature: 310.4,
      rotationalSpeed: 1340,
      torque: 69.4,
      toolWear: 205,
      deltaT: 10.9,
      powerWatts: 9737
    }
  },
  {
    id: 'PRED-1041',
    sampleId: 'SAMPLE-1928',
    equipmentName: 'High-Precision Grinder A',
    failureProbability: 0.386,
    failurePercentage: 38.6,
    riskTier: 'Elevated Wear',
    riskLevel: 'moderate',
    timestamp: '2026-10-24T22:30:00Z',
    modelVersion: 'rf-sklearn-demo-v1',
    isDemo: true,
    contributingFactors: [
      'Elevated tool wear (188 min); tool nearing end of optimal blade sharpness window'
    ],
    recommendedAction: 'Schedule physical inspection during the next planned maintenance window.',
    sensors: {
      airTemperature: 298.2,
      processTemperature: 308.6,
      rotationalSpeed: 1433,
      torque: 39.5,
      toolWear: 188,
      deltaT: 10.4,
      powerWatts: 5926
    }
  },
  {
    id: 'PRED-1040',
    sampleId: 'SAMPLE-1102',
    equipmentName: 'Vertical Machining Center 1',
    failureProbability: 0.051,
    failurePercentage: 5.1,
    riskTier: 'Optimal Operation',
    riskLevel: 'optimal',
    timestamp: '2026-10-24T20:10:00Z',
    modelVersion: 'rf-sklearn-demo-v1',
    isDemo: true,
    contributingFactors: [
      'All sensor telemetry parameters operate well within nominal baseline envelopes'
    ],
    recommendedAction: 'Routine monitoring active. No intervention required.',
    sensors: {
      airTemperature: 298.3,
      processTemperature: 308.8,
      rotationalSpeed: 1489,
      torque: 43.1,
      toolWear: 42,
      deltaT: 10.5,
      powerWatts: 6718
    }
  },
  {
    id: 'PRED-1039',
    sampleId: 'SAMPLE-0984',
    equipmentName: 'HMC Lathe Unit 02',
    failureProbability: 0.068,
    failurePercentage: 6.8,
    riskTier: 'Optimal Operation',
    riskLevel: 'optimal',
    timestamp: '2026-10-24T18:45:00Z',
    modelVersion: 'rf-sklearn-demo-v1',
    isDemo: true,
    contributingFactors: [
      'All sensor telemetry parameters operate well within nominal baseline envelopes'
    ],
    recommendedAction: 'Routine monitoring active. No intervention required.',
    sensors: {
      airTemperature: 298.2,
      processTemperature: 308.7,
      rotationalSpeed: 1408,
      torque: 46.3,
      toolWear: 28,
      deltaT: 10.5,
      powerWatts: 6825
    }
  },
  {
    id: 'PRED-1038',
    sampleId: 'SAMPLE-0751',
    equipmentName: 'Hydraulic Stamping Press 5',
    failureProbability: 0.912,
    failurePercentage: 91.2,
    riskTier: 'Critical Imminent Risk',
    riskLevel: 'critical',
    timestamp: '2026-10-24T16:15:20Z',
    modelVersion: 'rf-sklearn-demo-v1',
    isDemo: true,
    contributingFactors: [
      'Critical Torque-ToolWear overstrain index (15,358 > 11,000 threshold)',
      'Severe cutting tool wear degradation (232 min > 210 min replacement threshold)',
      'Extreme mechanical torque load (66.2 Nm exceeding high-strain limit of 62.0 Nm)',
      'Excessive mechanical spindle power (9,425 W > 8,500 W duty rating)'
    ],
    recommendedAction: 'Immediate intervention recommended. Halt cycle or dispatch maintenance engineer immediately.',
    sensors: {
      airTemperature: 301.0,
      processTemperature: 312.4,
      rotationalSpeed: 1360,
      torque: 66.2,
      toolWear: 232,
      deltaT: 11.4,
      powerWatts: 9425
    }
  }
];
