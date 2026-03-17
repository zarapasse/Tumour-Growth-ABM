"""This script runs and validates the Exponential ABM in a no-drug scenario."""

import json
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.utils_exponential import (
    process_config,
    run_abm,
    analytic_population_no_drug,
)

# ---------------------- Load config ---------------------- #
CONFIG_PATH = PROJECT_ROOT / "configs" / "exponential_config.json"
with CONFIG_PATH.open("r", encoding="utf-8") as f:
    config = json.load(f)

initial_cells, birth_rate, death_rate, dt, steps, n_runs, hill_params, _, _, _ = (
    process_config(config)
)
time = np.arange(steps + 1) * dt

# ---------------------- No-drug scenario ---------------------- #
scenarios = [{"name": "No drug", "schedule": [], "alpha": 0.0}]

# ---------------------- Run ABM ---------------------- #
abm_out = run_abm(config, scenarios, seed=42)

mean_abm = abm_out["mean_trajectories"][0]
std_abm = abm_out["std_trajectories"][0]

# ---------------------- Analytic solution ---------------------- #
N_det = analytic_population_no_drug(time, initial_cells, birth_rate, death_rate)

# ---------------------- Plot ---------------------- #
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(time, mean_abm, lw=2, label="ABM mean", color="blue")
ax.fill_between(
    time,
    mean_abm - std_abm,
    mean_abm + std_abm,
    alpha=0.25,
    label="ABM ± SD",
    color="blue",
)

ax.plot(time, N_det, linestyle="--", color="black", lw=2, label="Deterministic")

ax.set_xlabel("Time (days)")
ax.set_ylabel("Tumour population (cells)")
ax.set_title("Exponential ABM vs deterministic solution (no drug)")
ax.legend()
ax.grid(alpha=0.3)

# ---------------------- Parameter panel ---------------------- #
param_lines = [
    "Simulation parameters",
    "----------------------",
    f"initial_cells = {initial_cells}",
    f"birth_rate   = {birth_rate}",
    f"death_rate   = {death_rate}",
    f"dt           = {dt}",
    f"steps        = {steps}",
    f"n_runs       = {n_runs}",
]

param_text = "\n".join(param_lines)

fig.text(
    0.84,  # x-position
    0.95,  # y-position
    param_text,
    va="top",
    ha="left",
    fontsize=8,
    family="monospace",
    bbox=dict(
        boxstyle="round",
        facecolor="white",
        edgecolor="0.8",
        alpha=0.95,
    ),
)

plt.tight_layout(rect=(0, 0, 0.82, 1.0))
outpath = (
    PROJECT_ROOT
    / "results"
    / "exponential"
    / "no_resistance"
    / f"Exponential_validation.png"
)
outpath.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(outpath, dpi=300, bbox_inches="tight")
# plt.show()
plt.close(fig)

# ---------------------- Diagnostics ---------------------- #
print("\n--- No-drug validation ---")
print(f"Final Exponential ABM mean: {mean_abm[-1]:.2f}")
print(f"Final deterministic solution: {N_det[-1]:.2f}")
print(f"Relative error: {(mean_abm[-1] - N_det[-1]) / N_det[-1]:.2%}")
