# ABM_TUMOUR_GROWTH

Agent-based and deterministic tumour growth models for analysing **dosing schedule fragility** under exponential and logistic dynamics, with optional drug resistance.

This repository was developed as part of my MSci Mathematics dissertation investigating schedule sensitivity in stochastic tumour growth models, and their extension to include resistance.

---

## Structure

```
ABM_TUMOUR_GROWTH/
├── misc_analysis/      # PK models, Hill plots, data visualisation
└── src/
├── Exponential/    # Exponential ABM + resistance + fragility
└── Logistic/       # Resource-limited (logistic) ABM
```

### Agent-Based Models
Located in:
`
src/Exponential/
`
and 
`src/Logistic/`


Key scripts:
- `run_exponential.py` and `run_logistic`
- `run_exponential_no_drug.py` and `run_exponential_no_drug.py`
- `run_exponential_resistance.py` and `run_exponential_resistance.py`
-  Graphing scripts

Configuration files:
`config.json` and `config_logistic.json`

---

## Installation

From project root:

```bash
python -m venv venv
source venv/bin/activate
pip install numpy scipy matplotlib mesa
