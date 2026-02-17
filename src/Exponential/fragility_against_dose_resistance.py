"""
Fragility vs mean dose x_bar (WITH resistance)

Fragility metric:
    F = (V_odd - V_even) / V0

Compares:
- Deterministic fragility from analytic trajectories (PK + Hill kill, NO resistance)
- Resistant ABM fragility from stochastic simulations (mean ± 1 SD across runs)
"""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from utils import (
    make_fragility_test_scenarios,
    process_config,
    analytic_population_with_pk,
    run_abm_resistance,
)

# ----------------- Load config ----------------- #
CONFIG_PATH = Path(__file__).parent / "config.json"
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

# ----------------- Sweep settings ----------------- #
n_doses_per_cycle = 2
n_cycles = 4
cycle_length = 12
alpha_val = 1.0

t_end = n_cycles * cycle_length

dt = float(config["simulation"]["dt"])
steps = int(round(t_end / dt))  # number of step updates
config["simulation"]["steps"] = steps

(
    initial_cells,
    birth_rate,
    death_rate,
    dt,
    steps,
    n_runs,
    hill_params,
    p_mutation,
    initial_resistant_fraction,
    fitness_cost,
) = process_config(config)

time = np.arange(steps + 1) * dt

x_bar_values = np.arange(10, 100, 5, dtype=float)

# Storage
F_det = []
F_abm_mean = []
F_abm_std = []

# ----------------- Main loop ----------------- #
for x_bar in x_bar_values:
    total_dose_per_cycle = x_bar * n_doses_per_cycle

    sigma = (
        x_bar / 2.0
    )  #! Change to control variability of dose distribution across cycles

    scenarios = make_fragility_test_scenarios(
        total_dose_per_cycle,
        n_doses_per_cycle,
        n_cycles,
        sigma,
        cycle_length,
        alpha_val,
    )

    even_sc, odd_sc = scenarios[0], scenarios[1]

    # ---------- Deterministic fragility (NO resistance) ----------
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

    # ---------- Resistant ABM fragility ----------
    abm_out = run_abm_resistance(
        config,
        scenarios,
        seed=42,
        compute_fragility=True,
        enable_resistance=True,
    )

    per_run = np.array(abm_out["fragility"]["per_run"], dtype=float)
    F_abm_mean.append(float(per_run.mean()))
    F_abm_std.append(float(per_run.std(ddof=1)) if per_run.size > 1 else 0.0)

F_det = np.asarray(F_det, dtype=float)
F_abm_mean = np.asarray(F_abm_mean, dtype=float)
F_abm_std = np.asarray(F_abm_std, dtype=float)

# ----------------- Plot ----------------- #
fig, ax = plt.subplots(figsize=(10.5, 6.5))

ax.plot(x_bar_values, F_det, lw=2, label="Deterministic (no resistance)", color="black")
ax.plot(x_bar_values, F_abm_mean, lw=2, label="Resistant ABM mean", color="blue")
ax.fill_between(
    x_bar_values,
    F_abm_mean - F_abm_std,
    F_abm_mean + F_abm_std,
    alpha=0.15,
    label="Resistant ABM ± SD",
    color="blue",
)

ax.axhline(0.0, lw=1, ls="--")
ax.set_xlabel(r"Mean dose $\bar{x}$")
ax.set_ylabel("Fragility")
ax.set_title(f"Fragility vs mean dose (resistance ON, {n_cycles} cycles)")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9, ncol=2)

# ---------------------- Parameter side panel ---------------------- #
param_lines = [
    "Fragility sweep",
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
    f"p_mutation    = {p_mutation}",
    f"init_res_frac = {initial_resistant_fraction}",
    f"fitness_cost  = {fitness_cost}",
    "",
    "Hill parameters",
    "----------",
    f"K_kill = {hill_params.K_kill if hill_params is not None else None}",
    f"C      = {hill_params.C if hill_params is not None else None}",
    f"n      = {hill_params.n if hill_params is not None else None}",
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

outpath = (
    Path(__file__).parent
    / f"Graphs/Resistance/Fragility_vs_mean_dose_{n_cycles}_cycles_with_resistance.png"
)
plt.savefig(outpath, dpi=300, bbox_inches="tight")
#plt.show()
plt.close(fig)

print(f"Saved plot: {outpath}")
