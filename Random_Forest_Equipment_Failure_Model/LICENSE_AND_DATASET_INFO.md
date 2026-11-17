# Dataset Provenance, Sensor Specifications & Licensing Information

## 1. Dataset Provenance & Citation

The predictive maintenance model in this package was trained and validated on the **AI4I 2020 Predictive Maintenance Dataset**, created by **Stephan Matzka** (School of Engineering - Technology and Life, Hochschule für Technik und Wirtschaft Berlin, Germany).

**Official Citation:**
```bibtex
@misc{matzka2020ai4i,
  title        = {AI4I 2020 Predictive Maintenance Dataset},
  author       = {Matzka, Stephan},
  year         = {2020},
  howpublished = {UCI Machine Learning Repository},
  doi          = {10.24432/C5HS5C},
  url          = {https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset}
}
```

**License:**  
This dataset is licensed under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** license.  
You are free to share, copy, distribute, transmit, and adapt the dataset, provided appropriate credit is given.

---

## 2. Dataset Overview

The AI4I 2020 dataset reflects real-world predictive maintenance telemetry from an industrial milling machine. It contains **10,000 synthetic operating records** generated from mathematical equations governing thermodynamic heat transfer, tool wear degradation, motor torque-speed power curves, and stress overstrain limits.

- **Total Records:** 10,000
- **Overall Class Balance:** 9,661 Normal Operations (96.61%) vs. 339 Failure Events (3.39%)
- **Class Imbalance Ratio:** ~28.5 : 1

### How Teammates Can Obtain the Dataset
To keep this deployment package lightweight and clean, the full 10,000-row CSV is not bundled inside this ZIP. You can obtain `ai4i2020.csv` via:
1. **UCI Machine Learning Repository:**  
   Download from: `https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset`
2. **Kaggle Mirror:**  
   Search for "AI4I 2020 Predictive Maintenance Dataset" on Kaggle.
3. Once downloaded, place `ai4i2020.csv` in `data/raw/` or in the root directory before running the training scripts.

---

## 3. Sensor Features & Physical Specifications

The production Random Forest model receives **strictly 5 raw sensor inputs**:

| Sensor Feature | Physical Description | Units | Nominal Range | Sensor Type |
| :--- | :--- | :---: | :---: | :--- |
| **Air temperature [K]** | Ambient room temperature | Kelvin ($K$) | $295.3 - 304.5$ | Ambient Thermocouple |
| **Process temperature [K]** | Internal milling chamber temperature | Kelvin ($K$) | $305.7 - 313.8$ | Chamber RTD Sensor |
| **Rotational speed [rpm]** | Spindle angular rotational speed | Revolutions per minute ($rpm$) | $1,168 - 2,886$ | Optical Spindle Tachometer |
| **Torque [Nm]** | Mechanical torque exerted by spindle motor | Newton-meters ($Nm$) | $3.8 - 76.6$ | Strain Gauge Torque Transducer |
| **Tool wear [min]** | Cumulative cutting edge engagement time | Minutes ($min$) | $0 - 253$ | Machining Operation Log |

---

## 4. Internally Derived Domain-Physics Telemetry

Rather than forcing tree algorithms to approximate non-linear physical laws with hundreds of crude axis-aligned step splits, our pipeline internally derives 3 physics features:

### 1. Temperature Gradient ($\Delta T$)
$$\Delta T = \text{Process temperature} - \text{Air temperature} \quad [K]$$
- **Physical Rationale:** Detects **Heat Dissipation Failure (HDF)**. Heat generated during cutting must dissipate into ambient air. If $\Delta T < 8.6\text{ K}$ while spindle rotational speed is low ($\le 1,380\text{ rpm}$), heat buildup accelerates thermal seizure.

### 2. Spindle Mechanical Power ($P_{\text{mech}}$)
$$P_{\text{mech}} = \text{Torque} \times \left(\text{Rotational speed} \times \frac{2\pi}{60}\right) \quad [\text{Watts}]$$
- **Physical Rationale:** Detects **Power Failure (PWF)**. The milling motor stalls if cutting load draws power below $3,500\text{ W}$, or burns out if power exceeds $9,000\text{ W}$.

### 3. Overstrain Index ($OSI$)
$$OSI = \text{Torque} \times \text{Tool wear} \quad [Nm \cdot min]$$
- **Physical Rationale:** Detects **Overstrain Failure (OSF)**. As tool wear accumulates, cutting edge degradation elevates mechanical stress. If the product of torque and wear exceeds $11,000\text{ Nm}\cdot\text{min}$ ($12,000\text{ Nm}\cdot\text{min}$ for heavy tools), catastrophic structural breakage occurs.

---

## 5. Data Leakage Prevention Notice

The original dataset includes five post-incident diagnostic failure mode flags:
- `TWF`: Tool Wear Failure
- `HDF`: Heat Dissipation Failure
- `PWF`: Power Failure
- `OSF`: Overstrain Failure
- `RNF`: Random Failure

**CRITICAL RULE:** These flags represent post-failure investigation findings. In an active production deployment, they are unknown prior to equipment breakdown. Using them as inputs would constitute fatal data leakage. Our model **strictly excludes** them, predicting `Machine failure` purely from the 5 continuous physical sensors.
