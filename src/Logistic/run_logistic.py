import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path

from utils_logistic import (
    make_fragility_test_scenarios,
    process_scenarios_config,
    process_config,
    run_abm_logistic,
    deterministic_pk,
    logistic_fit,
)


# ---------------- Load Config ---------------- #
CONFIG_PATH = Path(__file__).parent / "config_logistic.json"
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

do_fragility_test = True


# ---------------- Build scenarios ---------------- #
if do_fragility_test:
    print("Running fragility test for logistic ABM...")

    x_bar = 25
    n_doses_per_cycle = 2
    total_dose_per_cycle = x_bar * n_doses_per_cycle
    sigma = total_dose_per_cycle / 2
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
    is_fragility = True

else:
    scenarios = process_scenarios_config(config)
    is_fragility = False


# Extract config parameters for baseline computation
initial_cells, birth_rate, death_rate, dt, steps, n_runs, hill_params = process_config(config)
time = np.arange(steps) * dt




# ---------------- Compute NO-DRUG baseline ---------------- #
print("\nComputing NO-DRUG baseline...")

baseline_scenario = [{
    "name": "No Drug",
    "schedule": [],
    "alpha": 0.0,
}]

baseline_results = run_abm_logistic(config, baseline_scenario, seed=42)

mean_no_drug = baseline_results["mean_trajectories"][0]
std_no_drug  = baseline_results["std_trajectories"][0]

# Fit logistic curve to no-drug mean trajectory
K0, r0, N0_0, baseline_fit = logistic_fit(time, mean_no_drug)

print(f"Baseline Logistic Fit: K={K0:.3f}, r={r0:.3f}, N0={N0_0:.1f}")





# ---------------- Run ABM ---------------- #
initial_cells, birth_rate, death_rate, dt, steps, n_runs, hill_params = process_config(config)
time = np.arange(steps) * dt


abm_results = run_abm_logistic(config, scenarios, seed=42, compute_fragility=True)

# ---------------- Collect results ---------------- #
results = []

for i, sc in enumerate(scenarios):

    mean_abm = abm_results["mean_trajectories"][i]
    std_abm  = abm_results["std_trajectories"][i]

    # analytic comparator: logistic fit of NO-DRUG baseline
    N_det = baseline_fit

    # deterministic PK curve
    conc_det = deterministic_pk(time, sc["schedule"], sc["alpha"])

    results.append({
        "name": sc["name"],
        "schedule": sc["schedule"],
        "alpha": sc["alpha"],
        "mean": mean_abm,
        "std": std_abm,
        "N_det": N_det,
        "conc_det": conc_det,
    })
    
    
    
    
    
    
# ------------------- Plotting ------------------- #
scenario_palette = [
    "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown",
    "tab:pink", "tab:gray", "tab:olive", "tab:cyan"
]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 10))


# ===================== TOP: Tumour Trajectories =====================
for i, res in enumerate(results):
    col = scenario_palette[i % len(scenario_palette)]

    ax1.plot(time, res["mean"], lw=2, color=col,
             label=f'{res["name"]} — ABM mean')

    ax1.fill_between(
        time,
        res["mean"] - res["std"],
        res["mean"] + res["std"],
        color=col,
        alpha=0.1,
    )

    # dosing lines
    for amt, t_dose in res["schedule"]:
        ax1.axvline(t_dose, color=col, ls=":", alpha=0.3)

ax1.set_title("Logistic Energy-based ABM — Tumour Trajectories")
ax1.set_ylabel("Cells")
ax1.grid(alpha=0.3)
ax1.legend(ncol=2, fontsize=9)


# ---- GLOBAL NO-DRUG ANALYTIC CURVE (black/grey) ---- #
ax1.plot(
    time,
    baseline_fit,
    "--",
    color="grey",
    lw=3,
    alpha=0.6,
    label="No-drug logistic fit"
)


# ===================== BOTTOM: PK Profiles =====================
for i, res in enumerate(results):
    col = scenario_palette[i % len(scenario_palette)]

    ax2.plot(
        time,
        res["conc_det"],
        lw=2,
        color=col,
        label=f'{res["name"]} (α={res["alpha"]})'
    )

ax2.set_title("PK Profiles")
ax2.set_xlabel("Time")
ax2.set_ylabel("Drug Concentration")
ax2.grid(alpha=0.3)
ax2.legend(ncol=2, fontsize=9)



# ----------------- parameter info box ----------------- #
param_lines = [
    f"initial_cells: {initial_cells}",
    f"birth_rate: {birth_rate}",
    f"death_rate: {death_rate}",
    f"dt: {dt}, steps: {steps}",
    f"n_runs: {n_runs}",
    "Hill parameters:",
    f"  E0: {hill_params['E0']}",
    f"  E1: {hill_params['E1']}",
    f"  C:  {hill_params['C']}",
    f"  n:  {hill_params['n']}",
]
param_text = "\n".join(param_lines)

# reserve a narrow area on the right for the slim info box and draw the text
plt.tight_layout(rect=(0, 0, 0.88, 1.0))   # leave ~12% on the right for the info box (moved left)
fig.text(
    0.855,                # moved left so the box sits just next to the axes
    0.98,                # start from top so lines flow downward
    param_text,
    fontsize=7,          # smaller font to fit the slim box
    va="top",
    ha="left",
    family="monospace",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="0.8"),
)









plt.tight_layout()
plt.show()


# ------------------------------------------------------
# FRAGILITY PRINT
# ------------------------------------------------------
if is_fragility:
    frag = abm_results["fragility"]
    print("\n===== FRAGILITY ANALYSIS (LOGISTIC ABM) =====")
    print(f"Mean fragility = {frag['mean']:.4f}")
    print(f"Per-run fragility = {frag['per_run']}")