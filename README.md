# AGENT-BASED MODELS OF TUMOUR GROWTH
## Pharmaceutical Dosing Schedule Sensitivity

This repository investigates how tumour response depends on **dosing schedule** under a range of mathematical and computational models.

This repository was developed as part of an MSci Mathematics dissertation at University College London (UCL), titled **_"Optimal Pharmaceutical Dosing Schedules in Stochastic Tumour Growth Models"_**.

We study how equal total drug doses, administered under different schedules (even vs uneven), can lead to **different treatment outcomes** due to nonlinear pharmacodynamics, ecological constraints, and evolutionary dynamics.

The analysis spans:
- Deterministic exponential growth
- Stochastic agent-based models (ABMs)
- Resource-limited (logistic) growth
- Emergent drug resistance

The goal is to understand when and why **treatment scheduling matters**, and how this depends on model structure.

---

## Models Included
### 1. Exponential Growth Models
- Deterministic baseline model
- Stochastic ABM implementation
- Schedule sensitivity emerges from nonlinear drug response

### 2. Logistic / Resource-Limited Models
- Mechanistic ABM with resource constraints
- Regime-dependent dynamics (growth vs saturation)
- Endpoint schedule sensitivity can disappear despite transient differences

### 3. Resistance Models
- Sensitive + resistant subpopulations
- Fitness cost of resistance
- Selection pressure induced by dosing schedules
  
---
## Metrics

Treatment response is evaluated using multiple metrics, since different summary measures can lead to different conclusions about dosing schedule performance.

### 1. Fragility

The primary fragility metric compares final tumour burden under uneven and even dosing schedules with equal mean dose:

```math
F(\bar{x},\sigma)
=
\frac{V_{\text{uneven}}(\bar{x},\sigma)-V_{\text{even}}(\bar{x},0)}{V_0}
```
where:
- `V_uneven(x̄, σ)` is the final tumour size under an uneven schedule  
- `V_even(x̄, 0)` is the final tumour size under the corresponding even schedule  
- `V₀` is the initial tumour volume
  
Interpretation:
- \(F > 0\): even dosing preferred  
- \(F < 0\): uneven dosing preferred
  
Captures schedule sensitivity at the **treatment endpoint**.

### 2. Area Under the Curve (AUC)

When endpoint tumour size fails to distinguish schedules (e.g. under resource limitation), cumulative tumour burden is measured via:

```math
\mathrm{AUC} = \int_0^T N(t)\,dt
```

and the corresponding AUC-based fragility metric is defined as:
```math
F_{\mathrm{AUC}}(\bar{x},\sigma)
=
\frac{\mathrm{AUC}(\bar{x},\sigma)-\mathrm{AUC}(\bar{x},0)}{\mathrm{AUC}_0}
```

Interpretation:
- `F_AUC > 0`: even dosing preferred  
- `F_AUC < 0`: uneven dosing preferred  

Captures **total tumour burden over time**, including transient dynamics.


### 3. Resistant Fraction

```math
f_R(t) = \frac{R(t)}{ (S(t) + R(t))}
```
Measures the proportion of resistant cells in the tumour.


## Repository Structure

```
ABM_TUMOUR_GROWTH/
├── configs/              # Model parameters
├── src/
│   ├── models/          # Core tumour models
│   ├── utils/           # Helper functions
│   └── experiments/     # Scripts for running simulations and creating analysis figures
├── results/             # Generated outputs
├── README.md
└── requirements.txt
```
---
## Configuration

All simulations are controlled via JSON config files:

```
configs/
├── exponential_config.json
└── logistic_config.json
```
These define:
- birth/death rates
- drug parameters (Hill function)
-	dosing schedules
- resistance parameters
- simulation settings

---

## Installation

```bash
git clone https://github.com/zarapasse/Tumour-Growth-ABM
cd Tumour-Growth-ABM
pip install -r requirements.txt
```


