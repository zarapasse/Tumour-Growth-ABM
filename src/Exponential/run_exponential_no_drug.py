import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from utils import (
    process_config,
    run_abm,
    analytic_population_no_drug,
)

# ---------------------- Load config ---------------------- #
CONFIG_PATH = Path(__file__).parent / "config.json"
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

initial_cells, birth_rate, death_rate, dt, steps, n_runs, hill_params = process_config(config)
time = np.arange(steps) * dt

# ---------------------- No-drug scenario ---------------------- #
scenarios = [
    {
        "name": "No drug",
        "schedule": [],
        "alpha": 0.0
    }
]

# ---------------------- Run ABM ---------------------- #
abm_out = run_abm(config, scenarios, seed=42)

mean_abm = abm_out["mean_trajectories"][0]
std_abm = abm_out["std_trajectories"][0]

# ---------------------- Analytic solution (no drug) ---------------------- #
N_det = analytic_population_no_drug(
    time,
    initial_cells,
    birth_rate,
    death_rate,
)

# ---------------------- Plot ---------------------- #
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(time, mean_abm, lw=2, label="Exponential ABM mean")
ax.fill_between(
    time,
    mean_abm - std_abm,
    mean_abm + std_abm,
    alpha=0.25,
    label="Exponential ABM ± SD"
)

ax.plot(
    time,
    N_det,
    "k--",
    lw=2,
    label="Deterministic solution",
)

ax.set_xlabel("Time")
ax.set_ylabel("Tumour cells")
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

# Leave space on the right for the panel
plt.tight_layout(rect=(0, 0, 0.82, 1.0))

fig.text(
    0.84,          # x-position (outside axes)
    0.95,          # y-position (top-aligned)
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
plt.savefig("validation_no_drug.png", dpi=300, bbox_inches="tight")
plt.show()

# ---------------------- Diagnostics ---------------------- #
print("\n--- No-drug validation ---")
print(f"Final Exponential ABM mean: {mean_abm[-1]:.2f}")
print(f"Final deterministic solution: {N_det[-1]:.2f}")
print(f"Relative error: {(mean_abm[-1] - N_det[-1]) / N_det[-1]:.2%}")