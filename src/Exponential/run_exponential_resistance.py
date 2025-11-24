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

    total_dose_per_cycle = 40
    n_doses_per_cycle = 2
    n_cycles = 4
    sigma = 20      # deviation from mean dose for uneven schedule
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

        all_runs = abm_out["all_trajectories"][0]  # shape (n_runs, steps)
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
fig = plt.figure(figsize=(14, 14))

if n_scenarios_with_resistance > 1:
    # Use GridSpec to create flexible layout
    gs = gridspec.GridSpec(3, n_scenarios_with_resistance, figure=fig)
    ax1 = fig.add_subplot(gs[0, :])  # Top row spans all columns
    ax2_list = [fig.add_subplot(gs[1, i]) for i in range(n_scenarios_with_resistance)]  # Middle row split
    ax3 = fig.add_subplot(gs[2, :])  # Bottom row spans all columns
else:
    # Standard 3x1 layout
    gs = gridspec.GridSpec(3, 1, figure=fig)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2_list = [fig.add_subplot(gs[1, 0])]
    ax3 = fig.add_subplot(gs[2, 0])

# ------------------ 1. Total population vs Analytic ------------------ #
for i, res in enumerate(results):
    col = scenario_palette[i % len(scenario_palette)]
    
    # ABM mean & std
    abm_mean_total = res["mean"].sum(axis=1) if res["mean"].ndim==2 else res["mean"] 
    abm_std_total = res["std"].sum(axis=1) if res["std"].ndim==2 else res["std"]
    
    ax1.plot(time, abm_mean_total, color=col, lw=2, label=f'{res["name"]} - ABM mean')
    ax1.fill_between(time, abm_mean_total - abm_std_total, abm_mean_total + abm_std_total,
                     color=col, alpha=0.1)
    
    # Deterministic analytic
    ax1.plot(time, res["N_det"], color=col, ls="--", lw=2, alpha=0.3,
             label=f'{res["name"]} - Analytic (no resistance)')
    
    # Dose times
    for amt, dose_t in res["schedule"]:
        ax1.axvline(dose_t, color=col, ls=":", alpha=0.3)

ax1.set_ylabel("Cells")
ax1.set_title("Tumour population: ABM vs Analytic")
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
    ax2_list[0].set_ylabel("Cells")
    ax2_list[0].set_title("Tumour composition")

# ------------------ 3. PK profiles (deterministic) ------------------ #
for i, res in enumerate(results):
    col = scenario_palette[i % len(scenario_palette)]
    ax3.plot(time, res["conc_det"], color=col, lw=2,
             label=f'{res["name"]} (α={res["alpha"]})')

ax3.set_xlabel("Time")
ax3.set_ylabel("Drug concentration")
ax3.set_title("PK profiles (drug schedules)")
ax3.grid(True, alpha=0.3)
ax3.legend(ncol=2, fontsize=9)

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

plt.tight_layout(rect=(0, 0, 0.88, 1.0))
fig.text(
    0.855,
    0.98,
    param_text,
    fontsize=7,
    va="top",
    ha="left",
    family="monospace",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="0.8"),
)
plt.savefig('exponential_resistance_results.png', dpi=300)
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
        
        