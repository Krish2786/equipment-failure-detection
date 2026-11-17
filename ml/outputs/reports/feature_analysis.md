# AI4I 2020 Predictive Maintenance - Sensor Telemetry & Feature Engineering Analysis

## 1. Primary Sensor Features Telemetry

### 1.1 Air Temperature $[K]$ and Process Temperature $[K]$
- **Physical Dynamics:** The ambient air temperature averages 300.00 K ($26.85^\circ$C) with a standard deviation of 2.00 K. The process temperature averages 310.01 K ($36.86^\circ$C) with a standard deviation of 1.48 K.
- **Thermodynamic Coupling:** The process temperature is thermodynamically coupled to air temperature via forced convective cooling ($r = +0.876$).
- **Failure Association:** In isolation, temperature alone has a moderate correlation with failure ($r \approx +0.05$ to $+0.04$), but the temperature differential is critical for Heat Dissipation Failures (HDF).

### 1.2 Rotational Speed $[rpm]$ and Spindle Torque $[Nm]$
- **Physical Dynamics:** Spindle speed averages 1,538.7 rpm (range: 1,168 to 2,886 rpm). Spindle torque averages 39.99 Nm (range: 3.8 to 76.6 Nm).
- **Inverse Power Relationship:** Rotational speed and torque exhibit a strong inverse physical correlation ($r = -0.875$), accurately modeling a spindle operating at roughly constant mechanical power:
$$P = \tau \cdot \omega = \tau \cdot \left(\frac{2 \pi N}{60}\right)$$
- **Failure Association:** Spindle stalls (high torque, low speed) trigger severe overstrain and power failure, while runaway speed at low torque triggers power cutoffs. Torque exhibits the highest individual Pearson correlation with failure ($r = +0.191$).

### 1.3 Tool Wear $[min]$
- **Physical Dynamics:** Uniformly distributed from 0 to 253 minutes (mean: 107.95 min, std: 63.65 min).
- **Degradation Physics:** Each machined workpiece adds tool wear depending on the alloy quality tier (High: +5 min, Medium: +3 min, Low: +2 min).
- **Failure Association:** Tool wear shows a positive correlation with failure ($r = +0.105$). Catastrophic tool failure (TWF) manifests primarily between 200 and 240 minutes.

---

## 2. Deep Dive: Physics-Based Engineered Features (Experimental)

To inform future model development without contaminating the baseline 5-feature requirement, three physics-informed features were mathematically derived and evaluated:

### 2.1 Temperature Differential ($\Delta T$)
$$\Delta T = \text{Process Temperature [K]} - \text{Air Temperature [K]}$$
- **Physical Basis:** Measures the thermal gradient available for passive and active convective cooling across the spindle bearing assembly.
- **Failure Mechanism (HDF):** If $\Delta T < 8.6$ K and rotational speed $N < 1,380$ rpm, heat dissipation fails and causes machine shutdown.
- **Statistical Metric:** Average $\Delta T = 10.00$ K. In failure records, $\Delta T$ frequently collapses below 9.0 K. Correlation with target: $r = -0.041$.

### 2.2 Mechanical Spindle Power ($P_\text{mech}$)
$$P_\text{mech} = \text{Torque [Nm]} \times \text{Rotational Speed [rpm]} \times \frac{2 \pi}{60} \quad \text{[Watts]}$$
- **Physical Basis:** Instantaneous mechanical work produced by the motor.
- **Failure Mechanism (PWF):** Safe operating boundaries are calibrated between **3,500 W and 9,000 W**. Power readings below 3,500 W (stalling/under-power) or above 9,000 W (overload) trigger Power Failure (PWF).
- **Statistical Metric:** Mean power = 6,439.4 W (std dev = 1,123.6 W, range: 1,192 W to 14,354 W). Power exhibits distinct multimodal failure modes outside normal operating limits.

### 2.3 Overstrain Index ($I_\text{overstrain}$)
$$I_\text{overstrain} = \text{Torque [Nm]} \times \text{Tool Wear [min]} \quad \text{[Nm} \cdot \text{min]}$$
- **Physical Basis:** Combined mechanical fatigue parameter. As cutting edge microgeometry degrades (higher tool wear), high torque produces excessive shear strain at the workpiece-tool interface.
- **Failure Mechanism (OSF):** Overstrain Failure occurs when $I_\text{overstrain}$ exceeds variant-specific thresholds:
  - Variant L: $> 11,000$ Nm$\cdot$min
  - Variant M: $> 12,000$ Nm$\cdot$min
  - Variant H: $> 13,000$ Nm$\cdot$min
- **Correlation with Target:** $r = +0.224$ (higher than any single raw sensor reading!).

---

## 3. Comparison of Primary Baseline vs Engineered Features

| Feature Configuration | Number of Inputs | Alignment with React Frontend | Predictive Advantage |
| :--- | :--- | :--- | :--- |
| **Primary Baseline (Prepared Now)** | **5 inputs** (`Air temp`, `Process temp`, `Rotational speed`, `Torque`, `Tool wear`) | **100% Direct Match** | Zero overhead; models learn non-linear boundaries directly via tree splitting. |
| **Experimental Augmented Set (Future Study)** | **8 inputs** (5 baseline + $\Delta T$, $P_\text{mech}$, $I_\text{overstrain}$) | Requires frontend calculation or backend derivation layer | Provides explicit linear boundaries corresponding to known physical failure tripwires. |
