"""
Fragility vs mean dose x_bar (WITH resistance)

This script plots the fragility metric
    F = (V_odd - V_even) / V0
as a function of mean dose per administration x_bar.

It compares:
- Deterministic fragility computed from analytic trajectories (with PK + Hill kill)
- ABM fragility computed from stochastic simulations (mean ± 1 SD across runs)

Schedules:
- Even: (x_bar, x_bar) each cycle
- Odd ("holiday"): (2*x_bar, 0) each cycle
  implemented via sigma = total_dose_per_cycle / 2, i.e. sigma = mean_dose.
"""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from utils import (
    make_fragility_test_scenarios,
    process_config,
    analytic_population_with_pk,
    run_abm_for_resistant,
)

# ----------------- Load config ----------------- #
CONFIG_PATH = Path(__file__).parent / "config.json"
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

initial_cells, birth_rate, death_rate, dt, steps, n_runs, hill_params = process_config(config)
time = np.arange(steps) * dt
# ----------------- Sweep settings (edit as needed) ----------------- #
n_doses_per_cycle = 2
n_cycles = 1
cycle_length = 12
alpha_val = 1.0

t_end = n_cycles * cycle_length

# IMPORTANT: update config first
dt = config["simulation"]["dt"]
config["simulation"]["steps"] = int(round(t_end / dt)) + 1

# Now re-read params AFTER updating config
initial_cells, birth_rate, death_rate, dt, steps, n_runs, hill_params = process_config(config)
time = np.arange(steps) * dt

x_bar_values = np.arange(10, 100, 5)

# Storage
F_det = []
F_abm_mean = []
F_abm_std = []

# ----------------- Main loop ----------------- #
for x_bar in x_bar_values:
    total_dose_per_cycle = x_bar * n_doses_per_cycle

    # "Holiday" / maximally uneven schedule:
    # mean_dose = total_dose_per_cycle / 2 = x_bar, sigma = x_bar
    # -> odd doses: (x_bar+sigma, x_bar-sigma) = (2x_bar, 0)
    sigma = total_dose_per_cycle / 2.0

    scenarios = make_fragility_test_scenarios(
        total_dose_per_cycle,
        n_doses_per_cycle,
        n_cycles,
        sigma,
        cycle_length,
        alpha_val,
    )

    # Scenarios are guaranteed by your utils to be:
    # scenarios[0] = Even Schedule, scenarios[1] = Odd Schedule
    even_sc = scenarios[0]
    odd_sc = scenarios[1]

    # ---------- Deterministic fragility ----------
    N_even, _ = analytic_population_with_pk(
        time,
        initial_cells,
        birth_rate,
        death_rate,
        even_sc["schedule"],
        even_sc["alpha"],
        hill_params,
    )
    N_odd, _ = analytic_population_with_pk(
        time,
        initial_cells,
        birth_rate,
        death_rate,
        odd_sc["schedule"],
        odd_sc["alpha"],
        hill_params,
    )

    V_even_det = float(N_even[-1])
    V_odd_det = float(N_odd[-1])
    F_det.append((V_odd_det - V_even_det) / float(initial_cells))

    # ---------- ABM fragility (no resistance) ----------
    abm_out = run_abm_for_resistant(config, scenarios, seed=42, compute_fragility=True)

    per_run = np.array(abm_out["fragility"]["per_run"], dtype=float)
    F_abm_mean.append(float(per_run.mean()))
    F_abm_std.append(float(per_run.std(ddof=1)) if per_run.size > 1 else 0.0)

F_det = np.array(F_det, dtype=float)
F_abm_mean = np.array(F_abm_mean, dtype=float)
F_abm_std = np.array(F_abm_std, dtype=float)

# ----------------- Plot ----------------- #
fig, ax = plt.subplots(figsize=(10.5, 6.5))

ax.plot(x_bar_values, F_det, lw=2, label="Deterministic", color='black')
ax.plot(x_bar_values, F_abm_mean, lw=2, label="Resistant ABM mean", color='blue')
ax.fill_between(
    x_bar_values,
    F_abm_mean - F_abm_std,
    F_abm_mean + F_abm_std,
    alpha=0.15,
    label="ABM ± SD",
    color='blue',
)

ax.axhline(0.0, lw=1, ls="--")
ax.set_xlabel(r"Mean dose $\bar{x}$")
ax.set_ylabel(r"Fragility")
ax.set_title("Fragility vs mean dose for Resistant ABM")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9, ncol=3)

# ---------------------- Parameter side panel ---------------------- #
param_lines = [
    "Fragility values",
    "--------------",
    f"alpha        = {alpha_val}",
    f"n_cycles     = {n_cycles}",
    f"cycle_length = {cycle_length}",
    f"doses/cycle  = {n_doses_per_cycle}",
    "",
    "Model parameters",
    "-----------",
    f"initial_cells = {initial_cells}",
    f"birth_rate    = {birth_rate}",
    f"death_rate    = {death_rate}",
    f"dt            = {dt}",
    f"steps         = {steps}",
    f"n_runs        = {n_runs}",
    f"p_mutation    = {config['simulation']['p_mutation']}",
    f"initial_resistant_fraction = {config['simulation']['initial_resistant_fraction']}",
    "",
    "Hill parameters",
    "----------",
    f"E0 = {hill_params['E0']}",
    f"E1 = {hill_params['E1']}",
    f"C  = {hill_params['C']}",
    f"n  = {hill_params['n']}",
]
param_text = "\n".join(param_lines)

plt.tight_layout(rect=(0, 0, 0.82, 1.0))
fig.text(
    0.84,
    0.95,
    param_text,
    va="top",
    ha="left",
    fontsize=8,
    family="monospace",
    bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.8", alpha=0.95),
)

outpath = Path(__file__).parent / "Fragility_vs_mean_dose_single_cycle_with_resistance.png"
plt.savefig(outpath, dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved plot: {outpath}")