"""
This script runs ABM simulations comparing drug dosing schedules
in the presence of drug resistance. It generates plots of tumour
trajectories, composition (sensitive vs resistant), and PK profiles.

Figure explicitly states parameters used for clarity.
"""


import matplotlib.pyplot as plt
from matplotlib import gridspec
import json
from pathlib import Path
import numpy as np

from utils import (
    make_fragility_test_scenarios,
    process_scenarios_config,
    process_config,
    analytic_population_with_pk,
    run_abm_for_resistant,
)

# Load parameters from JSON file
CONFIG_PATH = Path(__file__).parent / "config.json"
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

do_fragility_test = True

if do_fragility_test:
    print("Running fragility test scenarios...")

    # total_dose_per_cycle = 20
    # n_doses_per_cycle = 2
    # n_cycles = 4
    # sigma = 10      # deviation from mean dose for uneven schedule
    # cycle_length = 12
    # alpha_val = 1
    
    #! Hardcoded values for dosing analysis
    x_bar = 30
    n_doses_per_cycle = 2
    total_dose_per_cycle = x_bar * n_doses_per_cycle
    sigma = total_dose_per_cycle / 2    
    n_cycles = 4
    cycle_length = 12
    alpha_val = 1

    scenarios = make_fragility_test_scenarios(
        total_dose_per_cycle, n_doses_per_cycle, n_cycles, sigma, cycle_length, alpha_val
    )
    is_fragility = True
else:
    print("Processing scenarios from config file...")
    scenarios = process_scenarios_config(config)
    is_fragility = False

# ---------------- RUN ABM & ANALYTIC ----------------
results = []
per_run_fragility = None

initial_cells, birth_rate, death_rate, dt, steps, n_runs, hill_params = process_config(
    config
)
time = np.arange(steps) * dt

abm_all_runs = []  # store all trajectories for fragility calculation

if is_fragility:
    abm_results = run_abm_for_resistant(config, scenarios, seed=42, compute_fragility=True)

    # Extract mean/std for plotting
    results = []
    for i, sc in enumerate(scenarios):
        N_det, conc_det = analytic_population_with_pk(
            time,
            initial_cells,
            birth_rate,
            death_rate,
            sc["schedule"],
            sc["alpha"],
            hill_params,
        )
        results.append(
            {
                "name": sc["name"],
                "mean": abm_results["mean_trajectories"][i],
                "std": abm_results["std_trajectories"][i],
                "schedule": sc["schedule"],
                "alpha": sc["alpha"],
                "N_det": N_det,
                "conc_det": conc_det,
            }
        )


    # Extract fragility
    fragility_info = abm_results["fragility"]
    print(f"\n--- Fragility Analysis ---")
    print(f"Mean Fragility: {fragility_info['mean']:.4f}")
    print(f"Per-run Fragility: {fragility_info['per_run']}")

else:
    for sc in scenarios:
        abm_out = run_abm_for_resistant(config, [sc], seed=42)

        all_runs = abm_out["all_trajectories"][0] 
        abm_all_runs.append(all_runs)

        # Mean & std for plotting
        mean_abm = all_runs.mean(axis=0)
        std_abm = all_runs.std(axis=0)

        # Deterministic trajectory
        N_det, conc_det = analytic_population_with_pk(
            time,
            initial_cells,
            birth_rate,
            death_rate,
            sc["schedule"],
            sc["alpha"],
            hill_params,
        )

        results.append(
            {
                "name": sc["name"],
                "alpha": sc["alpha"],
                "schedule": sc["schedule"],
                "mean": mean_abm,
                "std": std_abm,
                "N_det": N_det,
                "conc_det": conc_det,
            }
        )

# ========== PLOTTING: Tumour trajectory, composition, and PK profiles ==========
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

# Count how many scenarios have resistance data
n_scenarios_with_resistance = sum(1 for res in results if res["mean"].ndim == 2)

# Create figure with appropriate layout
fig = plt.figure(figsize=(14, 10))

if n_scenarios_with_resistance > 1:
    # Use GridSpec to create flexible layout
    gs = gridspec.GridSpec(2, n_scenarios_with_resistance, figure=fig, height_ratios=[1, 1])
    ax1 = fig.add_subplot(gs[0, :])  # Top row spans all columns
    ax2_list = [fig.add_subplot(gs[1, i]) for i in range(n_scenarios_with_resistance)]  # Bottom row split
else:
    # Standard 2x1 layout
    gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1, 1])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2_list = [fig.add_subplot(gs[1, 0])]

# ------------------ 1. Total population vs Analytic ------------------ #
for i, res in enumerate(results):
    col = scenario_palette[i % len(scenario_palette)]
    
    # ABM mean & std
    abm_mean_total = res["mean"].sum(axis=1) if res["mean"].ndim==2 else res["mean"] 
    abm_std_total = res["std"].sum(axis=1) if res["std"].ndim==2 else res["std"]
    
    ax1.plot(time, abm_mean_total, color=col, lw=2, label=f'{res["name"]} - ABM')
    ax1.fill_between(time, abm_mean_total - abm_std_total, abm_mean_total + abm_std_total,
                     color=col, alpha=0.1)
    
    # Deterministic analytic
    ax1.plot(time, res["N_det"], color=col, ls="--", lw=2, alpha=0.3,
             label=f'{res["name"]} - Analytic (no resistance)')
    
    # Dose times
    for amt, dose_t in res["schedule"]:
        ax1.axvline(dose_t, color=col, ls=":", alpha=0.3)

ax1.set_ylabel("Population")
ax1.set_title("Tumour population: Resistant ABM vs Analytic")
ax1.grid(True, alpha=0.3)
ax1.legend(ncol=2, fontsize=9)

# ------------------ 2. Stacked plot: Sensitive vs Resistant ------------------ #
if n_scenarios_with_resistance > 0:
    subplot_idx = 0
    for res in results:
        if res["mean"].ndim == 2:  # shape (steps, 2)
            sens_mean = res["mean"][:, 0]
            res_mean = res["mean"][:, 1]
            
            ax2_list[subplot_idx].stackplot(time, sens_mean, res_mean, 
                                labels=["Sensitive", "Resistant"],
                                colors=["tab:blue", "tab:red"], alpha=0.7)
            ax2_list[subplot_idx].set_ylabel("Cells" if subplot_idx == 0 else "")
            ax2_list[subplot_idx].set_title(f'{res["name"]}: Sensitive vs Resistant')
            ax2_list[subplot_idx].grid(True, alpha=0.3)
            ax2_list[subplot_idx].legend(loc="upper right", fontsize=9)
            subplot_idx += 1
else:
    ax2_list[0].text(0.5, 0.5, "No resistance data available", 
             ha='center', va='center', transform=ax2_list[0].transAxes)
    ax2_list[0].set_ylabel("Population")
    ax2_list[0].set_title("Tumour composition")

# ---------------------- Resistance summary for side panel ---------------------- #
# (uses ABM mean trajectories; final fraction at last timepoint)


def time_to_resistance_dominance(time, mean_traj):
    """
    Returns the first time t such that Resistant > Sensitive
    using the ABM mean trajectory.

    If resistance never dominates, returns None.
    """
    if mean_traj.ndim != 2:
        return None

    sens = mean_traj[:, 0]
    res  = mean_traj[:, 1]

    idx = np.where(res > sens)[0]
    if len(idx) == 0:
        return None

    return time[idx[0]]

res_frac_lines = ["", "Resistance summary", "---------------------------"]

for res in results:
    if res["mean"].ndim == 2:
        final_sens = float(res["mean"][-1, 0])
        final_res  = float(res["mean"][-1, 1])
        final_tot  = final_sens + final_res
        final_frac = (final_res / final_tot) if final_tot > 0 else 0.0

        res_frac_lines.append(f"{res['name']}:")
        res_frac_lines.append(f"  Resistance Fraction = {final_frac:.3f}")
        res_frac_lines.append(f"  Resistant Cells     = {final_res:.0f}")
    else:
        res_frac_lines.append(f"{res['name']}: (no resistance split)")



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
    f"p_mutation   = {config['simulation']['p_mutation']}",
    f"initial_resistant_fraction = {config['simulation']['initial_resistant_fraction']}",
    "",
    "Hill parameters",
    "--------------",
    f"E0 = {hill_params['E0']}",
    f"E1 = {hill_params['E1']}",
    f"C  = {hill_params['C']}",
    f"n  = {hill_params['n']}",
    "",
    "Dosing parameters",
    "-----------------",
    f"x_bar            = {x_bar}",
    f"doses per cycle  = {n_doses_per_cycle}",
    f"sigma            = {sigma}",
    f"n_cycles         = {n_cycles}",
    f"cycle_length     = {cycle_length}",
    f"alpha            = {alpha_val}",
]+ res_frac_lines

param_text = "\n".join(param_lines)

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

plt.savefig('exponential_resistance_seeded.png', dpi=300)
plt.show()

# ----------------- print summary ----------------- #
print("\nSchedules compared:")
for res in results:
    if res["mean"].ndim == 2:
        final_sens = res["mean"][-1, 0]
        final_res = res["mean"][-1, 1]
        final_total = final_sens + final_res
        print(f'- {res["name"]}: doses={len(res["schedule"])} doses, alpha={res["alpha"]}')
        print(f'  Final ABM: {final_total:.0f} (Sensitive: {final_sens:.0f}, Resistant: {final_res:.0f})')
        print(f'  Analytic (no resistance): {res["N_det"][-1]:.0f}')
    else:
        final_total = res["mean"][-1]
        print(f'- {res["name"]}: doses={len(res["schedule"])} doses, alpha={res["alpha"]}')
        print(f'  Final ABM: {final_total:.0f}, Analytic: {res["N_det"][-1]:.0f}')

