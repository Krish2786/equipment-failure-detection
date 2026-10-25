import { getRiskTierDetails } from '../utils/formatters';
import { validateSensorReadings } from '../utils/validation';

let predictionCounter = 1050;

/**
 * Deterministic physics-based heuristic emulator for the Scikit-learn
 * Random Forest Failure Probability model (predict_proba).
 * Produces repeatable, physically grounded failure probabilities between 0.02 and 0.98.
 * 
 * @param {Object} sensorValues
 * @param {number} sensorValues.airTemperature - Ambient air temp in Kelvin [290-310]
 * @param {number} sensorValues.processTemperature - Process temp in Kelvin [300-320]
 * @param {number} sensorValues.rotationalSpeed - Spindle speed in rpm [1000-3000]
 * @param {number} sensorValues.torque - Torque in Nm [3-80]
 * @param {number} sensorValues.toolWear - Tool wear in minutes [0-260]
 * @returns {Promise<Object>} Diagnostic prediction output
 */
export async function predictEquipmentFailure(sensorValues) {
  // Simulate inference latency of an ML model pipeline (450ms)
  await new Promise((resolve) => setTimeout(resolve, 450));

  const validation = validateSensorReadings(sensorValues);
  if (!validation.isValid) {
    const firstError = Object.values(validation.errors)[0];
    throw new Error(`Sensor telemetry validation failed: ${firstError}`);
  }

  const airTemp = Number(sensorValues.airTemperature);
  const processTemp = Number(sensorValues.processTemperature);
  const speed = Number(sensorValues.rotationalSpeed);
  const torque = Number(sensorValues.torque);
  const toolWear = Number(sensorValues.toolWear);

  const deltaT = processTemp - airTemp;
  const powerWatts = torque * (speed * (2 * Math.PI / 60)); // P = tau * omega
  const overstrainFactor = torque * toolWear; // Overstrain Failure (OSF) correlation

  const contributingFactors = [];
  let stressScore = 0;

  // 1. Thermal Dissipation Stress (Heat Dissipation Failure - HDF)
  // Optimal deltaT is between 8.6 K and 11.5 K.
  if (deltaT < 8.6) {
    const diff = 8.6 - deltaT;
    const thermalStress = Math.min(diff * 0.22, 0.45);
    stressScore += thermalStress;
    contributingFactors.push(
      `Heat dissipation deficiency: Delta-T is low (${deltaT.toFixed(1)} K < 8.6 K nominal limit), causing internal heat accumulation`
    );
  } else if (deltaT > 12.0) {
    const diff = deltaT - 12.0;
    const thermalStress = Math.min(diff * 0.12, 0.25);
    stressScore += thermalStress;
    contributingFactors.push(
      `Elevated process thermal gradient: Delta-T is high (${deltaT.toFixed(1)} K > 12.0 K threshold), indicating cooling system strain`
    );
  }

  // Absolute thermal limit check
  if (processTemp >= 313.0) {
    stressScore += 0.15;
    contributingFactors.push(`Critical process temperature (${processTemp.toFixed(1)} K) near upper structural boundary`);
  }

  // 2. Mechanical Power & Overstrain Stress (Power Failure - PWF & OSF)
  // Continuous duty rated power is ~3,000 to 7,500 W
  if (powerWatts > 8500) {
    const pwrExcess = (powerWatts - 8500) / 3000;
    const pwrStress = Math.min(pwrExcess * 0.35, 0.45);
    stressScore += pwrStress;
    contributingFactors.push(
      `Excessive mechanical spindle power (${Math.round(powerWatts).toLocaleString()} W > 8,500 W duty rating)`
    );
  } else if (powerWatts < 2200 && torque > 45) {
    // High torque with abnormally low speed (near-stall condition)
    stressScore += 0.28;
    contributingFactors.push(`Spindle near-stall overload detected (High torque ${torque.toFixed(1)} Nm at low speed ${Math.round(speed)} rpm)`);
  }

  // Torque strain
  if (torque > 62.0) {
    const tqStress = Math.min(((torque - 62) / 18) * 0.32, 0.40);
    stressScore += tqStress;
    contributingFactors.push(`Extreme mechanical torque load (${torque.toFixed(1)} Nm exceeding high-strain limit of 62.0 Nm)`);
  }

  // Overstrain Failure (OSF): interaction of Torque * ToolWear
  if (overstrainFactor > 11000) {
    const osfExcess = Math.min(((overstrainFactor - 11000) / 5000) * 0.35, 0.45);
    stressScore += osfExcess;
    contributingFactors.push(
      `Critical Torque-ToolWear overstrain index (${Math.round(overstrainFactor).toLocaleString()} > 11,000 threshold)`
    );
  }

  // 3. Tool Wear Failure (TWF)
  // Cutting tool wear curve accelerates after 180 min, critical at >220 min
  if (toolWear >= 210) {
    const twExcess = ((toolWear - 210) / 50) * 0.42;
    stressScore += twExcess;
    contributingFactors.push(`Severe cutting tool wear degradation (${toolWear} min > 210 min replacement threshold)`);
  } else if (toolWear >= 175) {
    const twExcess = ((toolWear - 175) / 35) * 0.18;
    stressScore += twExcess;
    contributingFactors.push(`Elevated tool wear (${toolWear} min); tool nearing end of optimal blade sharpness window`);
  }

  // 4. Rotational Speed Extremes
  if (speed >= 2650) {
    stressScore += 0.18;
    contributingFactors.push(`High spindle velocity (${Math.round(speed)} rpm) inducing centrifugal bearing vibrations`);
  } else if (speed <= 1150 && torque >= 55) {
    stressScore += 0.20;
    contributingFactors.push(`High torque at low rotational velocity (${Math.round(speed)} rpm) risking motor stator overheating`);
  }

  // Base baseline nominal probability (physical baseline noise)
  const baseProbability = 0.035;

  // Logistic mapping function for smooth 0.02 - 0.98 probability
  // Score 0 -> ~3.5%
  // Score 0.4 -> ~25%
  // Score 0.8 -> ~50%
  // Score 1.3+ -> ~85%+
  const logisticProb = 1 / (1 + Math.exp(-2.8 * (stressScore - 0.75)));
  let finalProbability = Math.min(Math.max(logisticProb, baseProbability), 0.975);

  // If no negative factors were found, guarantee optimal low probability
  if (contributingFactors.length === 0) {
    finalProbability = 0.028 + (toolWear / 260) * 0.08 + ((torque - 3) / 80) * 0.04;
    contributingFactors.push('All sensor telemetry parameters operate well within nominal baseline envelopes');
    contributingFactors.push('Thermal dissipation rate, torque strain, and tool wear indices are within safety limits');
  }

  const failurePercentage = Number((finalProbability * 100).toFixed(1));
  const riskDetails = getRiskTierDetails(failurePercentage);
  predictionCounter += 1;

  return {
    id: `PRED-${predictionCounter}`,
    sampleId: `SAMPLE-${Math.floor(1000 + Math.random() * 9000)}`,
    failureProbability: Number(finalProbability.toFixed(3)),
    failurePercentage: failurePercentage,
    riskTier: riskDetails.tier,
    riskLevel: riskDetails.level,
    contributingFactors,
    recommendedAction: riskDetails.action,
    modelVersion: 'rf-sklearn-demo-v1',
    isDemo: true,
    timestamp: new Date().toISOString(),
    sensors: {
      airTemperature: airTemp,
      processTemperature: processTemp,
      rotationalSpeed: speed,
      torque: torque,
      toolWear: toolWear,
      deltaT: Number(deltaT.toFixed(1)),
      powerWatts: Math.round(powerWatts)
    }
  };
}
