import json
import copy
from pathlib import Path

import matplotlib.pyplot as plt

from utils_logistic import (
    make_fragility_test_scenarios,
    process_config,
    run_abm_logistic,
    continuum_logistic_with_pkpd,
    estimate_K_tail,
    fit_r_logit,
)

# ---------------- Load Config ---------------- #
CONFIG_PATH = Path(__file__).parent / "config_logistic.json"
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

# ---------------- Build scenarios ---------------- #
x_bar = 20
n_doses_per_cycle = 2
total_dose_per_cycle = x_bar * n_doses_per_cycle
sigma = x_bar / 2
n_cycles = 4
cycle_length = 12
alpha_val = 1

scenarios = make_fragility_test_scenarios(
    total_dose_per_cycle,
    n_doses_per_cycle,
    n_cycles,
    sigma,
    cycle_length,
    alpha_val,
)

# ---------------------- Run ABM ---------------------- #
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
) = process_config(config)

abm_results = run_abm_logistic(config, scenarios, seed=42)
time = abm_results["time"]

# ---------------------- Run ABM (no-drug baseline for r,K) ---------------------- #
print("\nComputing NO-DRUG baseline (for K)...")

baseline_scenario = [{"name": "No Drug", "schedule": [], "alpha": 0.0}]
cfg = copy.deepcopy(config)
cfg["simulation"]["n_runs"] = 10

# ---- 1) Estimate K from saturated no-drug run ----
K_fit_results = run_abm_logistic(cfg, baseline_scenario, seed=42)
time = K_fit_results["time"]
mean_no_drug = K_fit_results["mean_trajectories"][0]

K0 = estimate_K_tail(mean_no_drug, frac_tail=0.2)

# ---- 2) Estimate r from low-N no-drug run ----
cfg["simulation"]["initial_cells"] = 30

r_fit_results = run_abm_logistic(cfg, baseline_scenario, seed=42)

mean_lowN = r_fit_results["mean_trajectories"][0]
time_lowN = r_fit_results["time"]

r, _ = fit_r_logit(time_lowN, mean_lowN, K0, low_frac=0.2, high_frac=0.8)
print("K0 (saturated run):", K0)
print("r0 (low-N run):", r)

# ---------------------- Build deterministic + collect results ---------------------- #
results = []
for i, sc in enumerate(scenarios):
    mean_abm = abm_results["mean_trajectories"][i]
    std_abm = abm_results["std_trajectories"][i]

    N_det, conc_det, _, _ = continuum_logistic_with_pkpd(
        time=time,
        N0=mean_abm[0],
        K=K0,
        r=r,
        schedule=sc["schedule"],
        alpha=sc["alpha"],
        hill_params=hill_params,
    )

    results.append(
        {
            "name": sc["name"],
            "schedule": sc["schedule"],
            "alpha": sc["alpha"],
            "mean": mean_abm,
            "std": std_abm,
            "N_det": N_det,
            "conc_det": conc_det,
        }
    )


# ------------------- Plotting ------------------- #
scenario_palette = [
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
    "tab:olive",
    "tab:cyan",
]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 10))


# ===== TOP: Tumour trajectories =====
for i, res in enumerate(results):
    col = scenario_palette[i % len(scenario_palette)]

    ax1.plot(time, res["mean"], lw=2, color=col, label=f'{res["name"]} — ABM')
    ax1.fill_between(
        time,
        res["mean"] - res["std"],
        res["mean"] + res["std"],
        color=col,
        alpha=0.1,
    )

    ax1.plot(
        time,
        res["N_det"],
        ls="--",
        lw=2,
        color=col,
        alpha=0.5,
        label=f'{res["name"]} — logistic+PKPD',
    )

    for _, t_dose in res["schedule"]:
        ax1.axvline(t_dose, color=col, ls=":", alpha=0.25)

ax1.text(0.0, 1.02, r"$\mathbf{(A)}$", transform=ax1.transAxes, ha="left", va="bottom")
ax1.set_title("Tumour Population: Resource-Dependent ABM", fontsize=13)
ax1.set_xlabel("Time (days)")
ax1.set_ylabel("Tumour Population (cells)")
ax1.grid(alpha=0.3)
ax1.legend(ncol=2, fontsize=9)

# ===== BOTTOM: PK profiles =====
for i, res in enumerate(results):
    col = scenario_palette[i % len(scenario_palette)]
    ax2.plot(time, res["conc_det"], lw=2, color=col, label=f'{res["name"]}')

ax2.text(0.0, 1.02, r"$\mathbf{(B)}$", transform=ax2.transAxes, ha="left", va="bottom")
ax2.set_title("Pharmacokinetic Drug Concentration Profiles", fontsize=12)
ax2.set_xlabel("Time (days)")
ax2.set_ylabel("Drug Concentration (mg/L)")
ax2.grid(alpha=0.3)
ax2.legend(ncol=2, fontsize=9)

# ----------------- parameter info box ----------------- #
panel_lines = [
    "Sweep parameters",
    "----------------",
    f"alpha         = {alpha_val}",
    f"n_cycles      = {n_cycles}",
    f"cycle_length  = {cycle_length}",
    f"doses/cycle   = {n_doses_per_cycle}",
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

param_text = "\n".join(panel_lines)

plt.tight_layout(rect=(0, 0, 0.82, 1.0))
fig.text(
    0.84,
    0.98,
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
    / f"logistic_abm_trajectories_{initial_cells}_cells_{n_runs}_runs_{x_bar}.png"
)

out_path.parent.mkdir(parents=True, exist_ok=True)

plt.savefig(out_path, dpi=300)
plt.show()
# Close figure when running on command line
# plt.close()
