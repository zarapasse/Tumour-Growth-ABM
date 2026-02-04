import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path

from utils_logistic import (
    make_fragility_test_scenarios,
    process_scenarios_config,
    process_config,
    run_abm_logistic,
    logistic_fit,
    continuum_logistic_with_pkpd,
    set_panel_title,
)

# ---------------- Load Config ---------------- #
CONFIG_PATH = Path(__file__).parent / "config_logistic.json"
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

do_fragility_test = True


# ---------------- Build scenarios ---------------- #
if do_fragility_test:
    print("Running fragility test for logistic ABM...")

    x_bar = 20
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
    N_det, conc_det, conc_used, k_det = continuum_logistic_with_pkpd(
        time=time,
        dt=dt,
        N0=mean_no_drug[0],     # match ABM mean initial
        K=K0,
        r=r0,
        schedule=sc["schedule"],
        alpha=sc["alpha"],
        hill_params=hill_params,
        lag_one_step=True,      # IMPORTANT: match ABM
    )
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
             label=f'{res["name"]} — ABM')

    ax1.fill_between(
        time,
        res["mean"] - res["std"],
        res["mean"] + res["std"],
        color=col,
        alpha=0.1,
    )
    
    # continuum logistic + drug comparator
    ax1.plot(
        time,
        res["N_det"],
        ls="--",
        lw=2,
        color=col,
        alpha=0.4,
        label=f'{res["name"]} — logistic',
    )

    # dosing lines
    for amt, t_dose in res["schedule"]:
        ax1.axvline(t_dose, color=col, ls=":", alpha=0.3)
    

# Panel label (left-aligned)
ax1.set_title(r"$\mathbf{(A)}$", loc="left", fontsize=12, pad=6)

# Main title (centered)
ax1.set_title(
    "Tumour Population: Resource-Dependent ABM",
    loc="center",
    fontsize=13,
    pad=6,
)


ax1.set_xlabel("Time (days)")
ax1.set_ylabel("Tumour Population (cells)")
ax1.grid(alpha=0.3)
ax1.legend(ncol=2, fontsize=9)





# ===================== BOTTOM: PK Profiles =====================
for i, res in enumerate(results):
    col = scenario_palette[i % len(scenario_palette)]

    ax2.plot(
        time,
        res["conc_det"],
        lw=2,
        color=col,
        label=f'{res["name"]}'
    )


# Panel label (left-aligned)
ax2.set_title(r"$\mathbf{(B)}$", loc="left", fontsize=12, pad=6)

# Main title (centered)
ax2.set_title(
    "Pharmacokinetic Drug Concentration Profiles",
    loc="center",
    fontsize=12,
    pad=6,
)
ax2.set_xlabel("Time (days)")
ax2.set_ylabel("Drug Concentration (mg/L)")
ax2.grid(alpha=0.3)
ax2.legend(ncol=2, fontsize=9)


# ----------------- parameter info box ----------------- #
param_lines = [
    "Simulation parameters",
    "----------------------",
    f"initial_cells = {initial_cells}",
    f"birth_rate   = {birth_rate}",
    f"death_rate   = {death_rate}",
    f"dt           = {dt}",
    f"steps        = {steps}",
    f"n_runs       = {n_runs}",
    "Hill parameters:",
    "----------------------",
    f"  E0: {hill_params['E0']}",
    f"  E1: {hill_params['E1']}",
    f"  C:  {hill_params['C']}",
    f"  n:  {hill_params['n']}",
    "Logistic Fit (No Drug):",
    "----------------------",
    f"K:  {K0:.3f}",
    f"r:  {r0:.3f}",
    f"N0: {N0_0:.1f}",
]

param_text = "\n".join(param_lines)

# Reserve space on the right ONCE
plt.tight_layout(rect=(0, 0, 0.82, 1.0))

fig.text(
    0.84,          # x-position (outside axes)
    0.98,          # y-position (top-aligned)
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

plt.savefig(Path(__file__).parent / "logistic_abm_trajectories_continuum.png", dpi=300)
plt.show()
