"""
Fragility vs mean dose x_bar (LOGISTIC ABM, NO resistance) — ABM only

Fragility metric:
    F = (V_odd - V_even) / V0
as a function of mean dose per administration x_bar.

Compares:
- ABM fragility computed from stochastic simulations (mean ± 1 SD across runs)
"""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from utils_logistic import (
    process_config,
    make_fragility_test_scenarios,
    run_abm_logistic,
)

# ----------------- Load config ----------------- #
CONFIG_PATH = Path(__file__).parent / "config_logistic.json"
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

# ----------------- Sweep settings ----------------- #
n_doses_per_cycle = 2
n_cycles = 4
cycle_length = 12
alpha_val = 1.0

t_end = n_cycles * cycle_length

dt = float(config["simulation"]["dt"])
steps = int(round(t_end / dt))
config["simulation"]["steps"] = steps

(
    initial_cells,
    birth_rate,
    death_rate,
    dt,
    steps,
    n_runs,
    hill_params,
    res_params,
    initial_resources,
    initial_cell_energy,
    _,
    _,
    _,
) = process_config(config)

x_bar_values = np.arange(10, 100, 2.5)

# Store values
F_abm_mean = []
F_abm_std = []

# ----------------- Main loop ----------------- #
for x_bar in x_bar_values:
    total_dose_per_cycle = x_bar * n_doses_per_cycle
    sigma = x_bar / 2.0  # Taking variability of half mean dose

    scenarios = make_fragility_test_scenarios(
        total_dose_per_cycle=total_dose_per_cycle,
        n_doses_per_cycle=n_doses_per_cycle,
        n_cycles=n_cycles,
        sigma=sigma,
        cycle_length=cycle_length,
        alpha=alpha_val,
    )

    abm_out = run_abm_logistic(config, scenarios, seed=42, compute_fragility=True)

    per_run = np.asarray(abm_out["fragility"]["per_run"], dtype=float)
    F_abm_mean.append(float(per_run.mean()))
    F_abm_std.append(float(per_run.std(ddof=1)) if per_run.size > 1 else 0.0)

F_abm_mean = np.asarray(F_abm_mean, dtype=float)
F_abm_std = np.asarray(F_abm_std, dtype=float)

# ----------------- Plot ----------------- #
fig, ax = plt.subplots(figsize=(10.5, 6.5))

ax.plot(x_bar_values, F_abm_mean, lw=2, label="ABM mean", color="blue")
ax.fill_between(
    x_bar_values,
    F_abm_mean - F_abm_std,
    F_abm_mean + F_abm_std,
    alpha=0.15,
    label="ABM ± SD",
    color="blue",
)

ax.axhline(0.0, lw=1, ls="--")
ax.set_xlabel(r"Mean dose $\bar{x}$ (mg/L)")
ax.set_ylabel("Fragility")
ax.set_title(f"Fragility vs mean dose (Logistic ABM, {n_cycles} cycles)")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9)
# ---------------- Side parameter panel ---------------- #
panel_lines = [
    "Sweep parameters",
    "----------------",
    f"alpha         = {alpha_val}",
    f"n_cycles      = {n_cycles}",
    f"cycle_length  = {cycle_length}",
    f"doses/cycle   = {n_doses_per_cycle}",
    f"t_end         = {t_end}",
    "",
    "Model parameters",
    "-----------",
    f"initial_cells       = {initial_cells}",
    f"birth_rate          = {birth_rate}",
    f"death_rate          = {death_rate}",
    f"dt                  = {dt}",
    f"steps               = {steps}",
    f"n_runs              = {n_runs}",
    f"initial_resources   = {initial_resources}",
    f"initial_cell_energy = {initial_cell_energy}",
    "",
    "Resource parameters",
    "----------",
    f"Energy_capacity    = {res_params.energy_capacity}",
    f"resource_influx    = {res_params.resource_influx}",
    "",
    "Hill parameters",
    "----------",
    f"K_kill = {hill_params.K_kill}",
    f"C      = {hill_params.C}",
    f"n      = {hill_params.n}",
]
panel_text = "\n".join(panel_lines)

plt.tight_layout(rect=(0, 0, 0.82, 1.0))
fig.text(
    0.84,
    0.95,
    panel_text,
    va="top",
    ha="left",
    fontsize=8,
    family="monospace",
    bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.8", alpha=0.95),
)

outpath = (
    Path(__file__).parent
    / f"Graphs/No_Resistance/Fragility_vs_mean_dose_logistic_{n_cycles}_cycles_Regime_C.png"
)
outpath.parent.mkdir(parents=True, exist_ok=True)

plt.savefig(outpath, dpi=300, bbox_inches="tight")
# plt.show()
# Close figure when running on command line
plt.close()
