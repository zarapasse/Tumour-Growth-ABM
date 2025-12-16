"""
Run ABM simulations (averaged over n_runs) over a sweep of mean doses x_bar.

For each x_bar:
- Build even/odd schedules (using make_fragility_test_scenarios)
- Run n_runs stochastic realisations
- Save mean ± std time series to CSV
- Save full parameter provenance to JSON
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from Exponential_Model_Complete import TumourModel
from utils import process_config, make_fragility_test_scenarios


# -------------------- CONFIG -------------------- #

CONFIG_PATH = Path(__file__).parent / "config.json"
OUTPUT_DIR = Path(__file__).parent / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ENABLE_RESISTANCE = True
P_MUTATION = 1e-4
SEED = 42

# Treatment structure (fixed across the sweep)
N_DOSES_PER_CYCLE = 2
N_CYCLES = 4
CYCLE_LENGTH = 12
ALPHA = 1.0

# Sweep settings: mean dose per administration (x_bar)
X_BAR_VALUES = list(range(10, 51, 10))  # 10, 20, ..., 50

# Sigma policy: sigma = f(x_bar) (treatment holidays)
def sigma_from_xbar(x_bar: float) -> float:
    return x_bar


# -------------------- LOAD BASE PARAMS -------------------- #

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

(
    initial_cells,
    birth_rate,
    death_rate,
    dt,
    steps,
    n_runs,
    hill_params,
) = process_config(config)

np.random.seed(SEED)


# -------------------- CORE RUNNER -------------------- #

def run_abm_average_for_scenario(scenario: dict, x_bar: float, sigma: float):
    """
    Run n_runs ABM realisations for a single scenario dict and save mean ± std + params.
    """

    sensitive_all = []
    resistant_all = []
    total_all = []
    resistant_frac_all = []

    for r in range(n_runs):
        model = TumourModel(
            initial_cells=initial_cells,
            birth_rate=birth_rate,
            death_rate=death_rate,
            dt=dt,
            alpha=scenario["alpha"],
            drug_schedule=scenario["schedule"],
            hill_params=hill_params,
            enable_resistance=ENABLE_RESISTANCE,
            p_mutation=P_MUTATION,
        )

        for _ in range(steps):
            model.step()

        df = model.datacollector.get_model_vars_dataframe()

        sensitive = df["Sensitive"].values[:steps]
        resistant = df["Resistant"].values[:steps]
        total = df["Total"].values[:steps]

        sensitive_all.append(sensitive)
        resistant_all.append(resistant)
        total_all.append(total)
        resistant_frac_all.append(resistant / np.maximum(1, total))

    sensitive_all = np.array(sensitive_all)
    resistant_all = np.array(resistant_all)
    total_all = np.array(total_all)
    resistant_frac_all = np.array(resistant_frac_all)

    time = np.arange(steps) * dt

    out_df = pd.DataFrame({
        "time": time,

        "Sensitive_mean": sensitive_all.mean(axis=0),
        "Sensitive_std": sensitive_all.std(axis=0),

        "Resistant_mean": resistant_all.mean(axis=0),
        "Resistant_std": resistant_all.std(axis=0),

        "Total_mean": total_all.mean(axis=0),
        "Total_std": total_all.std(axis=0),

        "ResistantFraction_mean": resistant_frac_all.mean(axis=0),
        "ResistantFraction_std": resistant_frac_all.std(axis=0),

        # PK deterministic for given schedule
        "DrugConc": df["DrugConc"].values[:steps],
    })

    schedule_type = "even" if "even" in scenario["name"].lower() else "odd"
    base_name = f"{schedule_type}_x{int(x_bar):03d}"

    csv_path = OUTPUT_DIR / f"{base_name}.csv"
    out_df.to_csv(csv_path, index=False)
    print(f"Saved averaged results: {csv_path}")

    params = {
        "model": "Exponential ABM with resistance",
        "scenario": scenario["name"],
        "schedule_type": schedule_type,

        "simulation": {
            "initial_cells": initial_cells,
            "birth_rate": birth_rate,
            "death_rate": death_rate,
            "dt": dt,
            "steps": steps,
            "n_runs": n_runs,
        },

        "drug": {
            "mean_dose_per_administration_x_bar": x_bar,
            "sigma": sigma,
            "alpha": scenario["alpha"],
            "cycle_length": CYCLE_LENGTH,
            "n_cycles": N_CYCLES,
            "n_doses_per_cycle": N_DOSES_PER_CYCLE,
            "total_dose_per_cycle": x_bar * N_DOSES_PER_CYCLE,
            "schedule": scenario["schedule"],
        },

        "pharmacodynamics": hill_params,

        "resistance": {
            "enabled": ENABLE_RESISTANCE,
            "p_mutation": P_MUTATION,
        },

        "random_seed": SEED,
        "timestamp": datetime.now().isoformat(),
    }

    json_path = OUTPUT_DIR / f"{base_name}_params.json"
    with open(json_path, "w") as f:
        json.dump(params, f, indent=2)
    print(f"Saved parameters: {json_path}")


# -------------------- MAIN SWEEP -------------------- #

if __name__ == "__main__":

    for x_bar in X_BAR_VALUES:
        sigma = sigma_from_xbar(x_bar)

        total_dose_per_cycle = x_bar * N_DOSES_PER_CYCLE

        scenarios = make_fragility_test_scenarios(
            total_dose=total_dose_per_cycle,
            n_doses=N_DOSES_PER_CYCLE,
            n_cycles=N_CYCLES,
            sigma=sigma,
            cycle_length=CYCLE_LENGTH,
            alpha=ALPHA,
        )

        print(f"\n=== x_bar={x_bar} | sigma={sigma} | total_dose_per_cycle={total_dose_per_cycle} ===")

        for sc in scenarios:  # Even + Odd
            run_abm_average_for_scenario(sc, x_bar=x_bar, sigma=sigma)

    print("\nDose sweep completed.")