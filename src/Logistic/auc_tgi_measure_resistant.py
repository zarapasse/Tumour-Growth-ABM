"""
AUC-based fragility sweep (WITH resistance)

AUC = ∫_0^T N_tot(t) dt, where N_tot = N_sensitive + N_resistant.

Transient fragility:
    F_AUC = (AUC_odd - AUC_even) / AUC0
where AUC0 is the no-treatment baseline AUC over the same horizon.

Computes:
- ABM mean ± SD of F_AUC across stochastic runs
"""

import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt

from utils_logistic import (
    run_abm_resistant_logistic,
    make_fragility_test_scenarios,
    process_config,
    compute_auc,
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
config["simulation"]["steps"] = steps  # overwrite horizon

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
    p_mutation,
    initial_resistant_fraction,
    fitness_cost,
) = process_config(config)

x_bar_values = np.arange(10, 100, 2.5)

# Store values
F_AUC_mean = []
F_AUC_std = []

# ---------------- Baseline no-drug AUC0 (same horizon) ---------------- #
baseline = [{"name": "No drug", "schedule": [], "alpha": 0.0}]
base_out = run_abm_resistant_logistic(config, baseline, seed=42)

base_trajs_total = base_out["all_total"][0]  # shape (n_runs, steps+1)
AUC0 = compute_auc(base_trajs_total, dt)  # shape (n_runs,)
AUC0_mean = float(AUC0.mean())

# ----------------- Main loop ----------------- #
for x_bar in x_bar_values:
    total_dose_per_cycle = x_bar * n_doses_per_cycle
    sigma = x_bar / 2.0

    scenarios = make_fragility_test_scenarios(
        total_dose_per_cycle=total_dose_per_cycle,
        n_doses_per_cycle=n_doses_per_cycle,
        n_cycles=n_cycles,
        sigma=sigma,
        cycle_length=cycle_length,
        alpha=alpha_val,
    )

    abm_out = run_abm_resistant_logistic(config, scenarios, seed=42)

    even_trajs_total = abm_out["all_total"][0]  # (n_runs, steps+1)
    odd_trajs_total = abm_out["all_total"][1]

    AUC_even = compute_auc(even_trajs_total, dt)  # (n_runs,)
    AUC_odd = compute_auc(odd_trajs_total, dt)

    F_AUC = (AUC_odd - AUC_even) / AUC0

    F_AUC_mean.append(float(F_AUC.mean()))
    F_AUC_std.append(float(F_AUC.std(ddof=1)) if F_AUC.size > 1 else 0.0)

F_AUC_mean = np.asarray(F_AUC_mean, dtype=float)
F_AUC_std = np.asarray(F_AUC_std, dtype=float)

# ------------ Plot ------------------ #
fig, ax = plt.subplots(figsize=(12, 7))

ax.plot(x_bar_values, F_AUC_mean, lw=2, label="ABM mean", color="blue")
ax.fill_between(
    x_bar_values,
    F_AUC_mean - F_AUC_std,
    F_AUC_mean + F_AUC_std,
    alpha=0.15,
    label="ABM ± SD",
    color="blue",
)
ax.axhline(0, ls="--", lw=1)

ax.set_title(f"AUC-based fragility vs mean dose (resistant ABM — {n_cycles} cycles)")
ax.set_xlabel(r"Mean dose $\bar{x}$ (mg/L)")
ax.set_ylabel("AUC fragility")
ax.grid(True, alpha=0.3)
ax.legend()

# ---------------- Side parameter panel ---------------- #
panel_lines = [
    "Sweep parameters",
    "----------------",
    f"alpha         = {alpha_val}",
    f"n_cycles      = {n_cycles}",
    f"cycle_length  = {cycle_length}",
    f"t_end         = {t_end}",
    "",
    "Model parameters",
    "----------------",
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
    "-------------------",
    f"energy_capacity   = {res_params.energy_capacity}",
    f"resource_influx   = {res_params.resource_influx}",
    "",
    "Resistance",
    "----------",
    f"p_mutation             = {p_mutation}",
    f"initial_resistant_frac = {initial_resistant_fraction}",
    f"fitness_cost           = {fitness_cost}",
    "",
    "Hill parameters",
    "---------------",
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
    / f"Graphs/Resistance/auc_fragility_logistic_{n_cycles}_cycles_Regime_C.png"
)
outpath.parent.mkdir(parents=True, exist_ok=True)

plt.savefig(outpath, dpi=300, bbox_inches="tight")
# plt.show()
plt.close()
