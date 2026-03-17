# TODO: fix this for the new code!!

import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.utils_logistic import (
    make_fragility_test_scenarios,
    process_config,
    run_abm_logistic,
    run_abm_resistant_logistic,
    pk_conc,
    plot_composition,
)

# ---------------- Load Config ---------------- #
CONFIG_PATH = PROJECT_ROOT / "configs" / "logistic_config.json"
with CONFIG_PATH.open("r", encoding="utf-8") as f:
    config = json.load(f)

print("Running fragility test scenarios...")

# ---------------------- Scenario parameters ---------------------- #
x_bar = 20
n_doses_per_cycle = 2
total_dose_per_cycle = x_bar * n_doses_per_cycle
sigma = x_bar / 2
n_cycles = 4
cycle_length = 12
alpha_val = 1.0

scenarios = make_fragility_test_scenarios(
    total_dose_per_cycle,
    n_doses_per_cycle,
    n_cycles,
    sigma,
    cycle_length,
    alpha_val,
)

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

# ---------------------- Run ABMs ---------------------- #
print("\nRunning normal ABM...")
abm_normal = run_abm_logistic(config, scenarios, seed=42)
time = abm_normal["time"]

print("Running resistant ABM...")
abm_resist = run_abm_resistant_logistic(config, scenarios, seed=42)

# ---------------------- Results ---------------------- #
results = []
for i, sc in enumerate(scenarios):
    normal_mean = abm_normal["mean_trajectories"][i]
    normal_std = abm_normal["std_trajectories"][i]

    resist_total_mean = abm_resist["mean_total"][i]
    resist_total_std = abm_resist["std_total"][i]

    results.append(
        {
            "name": sc["name"],
            "schedule": sc["schedule"],
            "alpha": sc.get("alpha", None),
            "normal_mean": normal_mean,
            "normal_std": normal_std,
            "resist_total_mean": resist_total_mean,
            "resist_total_std": resist_total_std,
            "mean_sensitive": abm_resist["mean_sensitive"][i],
            "std_sensitive": abm_resist["std_sensitive"][i],
            "mean_resistant": abm_resist["mean_resistant"][i],
            "std_resistant": abm_resist["std_resistant"][i],
        }
    )


# ---------------------- Helpers ---------------------- #
def time_to_resistance_dominance(time, sens, res):
    """First time t such that R(t) > S(t) in the mean trajectories."""
    idx = np.where(res > sens)[0]
    return float(time[idx[0]]) if len(idx) else None


# ---------------------- Plotting ---------------------- #
scenario_palette = ["tab:orange", "tab:green", "tab:red", "tab:purple"]

# Layout: A, (B,C), D + right panel column
fig = plt.figure(figsize=(15, 10))
gs = gridspec.GridSpec(
    3,
    3,
    width_ratios=[1.0, 1.0, 0.55],  # right column = panel
    height_ratios=[1.15, 1.0, 0.85],  # A, B/C, D
    hspace=0.40,
    wspace=0.30,
)

axA = fig.add_subplot(gs[0, 0:2])
axB = fig.add_subplot(gs[1, 0])
axC = fig.add_subplot(gs[1, 1])
axD = fig.add_subplot(gs[2, 0:2])

axP = fig.add_subplot(gs[:, 2])
axP.axis("off")

# (A) totals: resistant vs normal
for i, res in enumerate(results):
    col = scenario_palette[i % len(scenario_palette)]

    axA.plot(
        time,
        res["resist_total_mean"],
        color=col,
        lw=2,
        label=f'{res["name"]} — Resistant ABM',
    )
    axA.fill_between(
        time,
        res["resist_total_mean"] - res["resist_total_std"],
        res["resist_total_mean"] + res["resist_total_std"],
        color=col,
        alpha=0.08,
    )

    axA.plot(
        time,
        res["normal_mean"],
        color=col,
        lw=2,
        ls="--",
        alpha=0.85,
        label=f'{res["name"]} — Normal ABM',
    )
    axA.fill_between(
        time,
        res["normal_mean"] - res["normal_std"],
        res["normal_mean"] + res["normal_std"],
        color=col,
        alpha=0.12,
    )

    for _, dose_t in res["schedule"]:
        axA.axvline(dose_t, color=col, ls=":", alpha=0.25)

axA.set_title(r"$\mathbf{(A)}$  Total tumour population", loc="left", fontsize=12)
axA.set_ylabel("Cells")
axA.grid(True, alpha=0.3)
axA.legend(ncol=2, fontsize=9)


# ------------------- MIDDLE: composition stackplots ------------------- #
plot_composition(
    axB, results[0], r"$\mathbf{(B)}$  Even schedule: Sensitive vs Resistant", time
)
plot_composition(
    axC, results[1], r"$\mathbf{(C)}$  Odd schedule: Sensitive vs Resistant", time
)

# (D) PK profiles

# ------------------- BOTTOM: PK profiles ------------------- #
for i, sc in enumerate(scenarios):
    col = scenario_palette[i % len(scenario_palette)]
    conc = pk_conc(time, sc["schedule"], float(sc["alpha"]))
    axD.plot(time, conc, color=col, lw=2, label=sc["name"])

axD.set_title(r"$\mathbf{(D)}$  PK profiles", loc="left", fontsize=12)
axD.set_xlabel("Time (days)")
axD.set_ylabel("Drug concentration")
axD.grid(True, alpha=0.3)
axD.legend(fontsize=9, ncol=2)

# ---------------------- Right panel ---------------------- #
param_lines = [
    "Simulation Parameters",
    "---------------------",
    f"initial_cells = {initial_cells}",
    f"birth_rate    = {birth_rate}",
    f"death_rate    = {death_rate}",
    f"dt            = {dt}",
    f"steps         = {steps}",
    f"n_runs        = {n_runs}",
    "",
    "Resource Parameters",
    "-------------------",
    f"energy_capacity  = {res_params.energy_capacity}",
    f"div_threshold    = {res_params.division_threshold}",
    f"maintenance_cost = {res_params.maintenance_cost}",
    f"resource_influx  = {res_params.resource_influx}",
    "",
    "Resistance Parameters",
    "-------------------",
    f"p_mutation              = {p_mutation}",
    f"initial_resistant_frac  = {initial_resistant_fraction}",
    f"fitness_cost            = {fitness_cost}",
    "",
    "Hill Parameters",
    "---------------",
    f"K_kill = {hill_params.K_kill}",
    f"C      = {hill_params.C}",
    f"n      = {hill_params.n}",
]

param_text = "\n".join(param_lines)

axP.text(
    0.0,
    1.0,
    "\n".join(param_lines),
    va="top",
    ha="left",
    fontsize=8,
    family="monospace",
    bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.8", alpha=0.95),
)

fig.tight_layout()

outpath = (
    PROJECT_ROOT
    / "results"
    / "logistic"
    / "resistance"
    / f"logistic_normal_vs_resistant.png"
)
outpath.parent.mkdir(parents=True, exist_ok=True)

plt.savefig(outpath, dpi=300, bbox_inches="tight")
# plt.show()
plt.close(fig)

print(f"Saved plot: {outpath}")
