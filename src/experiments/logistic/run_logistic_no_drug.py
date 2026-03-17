import numpy as np
import matplotlib.pyplot as plt
import sys

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.utils_logistic import (
    process_config,
    fit_logistic_direct,
    run_abm_logistic,
)

# ---------------------- Load config ---------------------- #
CONFIG_PATH = PROJECT_ROOT / "configs" / "logistic_config.json"
with CONFIG_PATH.open("r", encoding="utf-8") as f:
    config = json.load(f)

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
time = np.arange(steps + 1) * dt

# ---------------------- No-drug scenario ---------------------- #
scenarios = [{"name": "No drug", "schedule": [], "alpha": 0.0}]

# ---------------------- Run ABM ---------------------- #
abm_out = run_abm_logistic(config, scenarios, seed=42)

mean_abm = abm_out["mean_trajectories"][0]
std_abm = abm_out["std_trajectories"][0]

# ---------------------- Logistic fit ---------------------- #
K1, r1, N0_1, fit1 = fit_logistic_direct(time, mean_abm)

# ---------------------- Plot ---------------------- #
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(time, mean_abm, lw=2, label="Mean cell count", color="blue")
ax.fill_between(
    time,
    mean_abm - std_abm,
    mean_abm + std_abm,
    alpha=0.25,
    label="Mean ± SD",
    color="blue",
)

ax.plot(time, fit1, "k--", lw=2, label="Logistic fit")

ax.set_xlabel("Time")
ax.set_ylabel("Tumour Population (cells)")
ax.set_title("Resource-dependent ABM vs Logistic Fit (no drug)")
ax.legend()
ax.grid(alpha=0.3)

# ---------------------- Parameter panel ---------------------- #
param_lines = [
    "Simulation parameters",
    "----------------------",
    f"steps            = {steps}",
    f"dt               = {dt}",
    f"n_runs            = {n_runs}",
    f"initial_cells     = {initial_cells}",
    "",
    "Model parameters",
    "----------------------",
    f"birth_rate        = {birth_rate}",
    f"death_rate        = {death_rate}",
    f"initial_resources = {initial_resources}",
    f"initial_energy    = {initial_cell_energy}",
    "",
    "Resource parameters",
    "----------------------",
    f"resource_influx   = {res_params.resource_influx}",
    f"energy_capacity   = {res_params.energy_capacity}",
    f"division_thresh   = {res_params.division_threshold}",
    f"maintenance_cost  = {res_params.maintenance_cost}",
    "",
    "Logistic fit",
    "----------------------",
    f"K   = {K1:.3g}",
    f"r   = {r1:.3g}",
    f"N0  = {N0_1:.3g}",
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

out_path = (
    Path(__file__).parent
    / "Graphs"
    / "No_Resistance"
    / f"logistic_no_drug_{initial_cells}_{n_runs}.png"
)

out_path.parent.mkdir(parents=True, exist_ok=True)

plt.tight_layout(rect=(0, 0, 0.82, 1.0))
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.show()
# Close figure when running on command line
# plt.close()
