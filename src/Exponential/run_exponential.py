import matplotlib.pyplot as plt
import json
from pathlib import Path
import numpy as np

from utils import (
    make_fragility_test_scenarios,
    process_scenarios_config,
    process_config,
    analytic_population_with_pk,
    run_abm,
    calculate_fragility,
)

# Load parameters from JSON file
CONFIG_PATH = Path(__file__).parent / "config.json"
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

do_fragility_test = True

if do_fragility_test:
    print("Running fragility test scenarios...")

    total_dose = 20
    n_doses = 2
    n_cycles = 2
    epsilon = 0.5
    cycle_length = 10
    alpha_val = 0.5

    scenarios = make_fragility_test_scenarios(
        total_dose, n_doses, n_cycles, epsilon, cycle_length, alpha_val
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
    abm_results = run_abm(config, scenarios, seed=42, compute_fragility=True)

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
        abm_out = run_abm(config, [sc], seed=42)

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


# ========== PLOTTING: Tumour trajectory and PK profiles across scenarios ==========
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

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 10), sharex=False)

# Population comparison (only drug schedules)
for i, res in enumerate(results):
    col = scenario_palette[i % len(scenario_palette)]
    ax1.plot(time, res["mean"], color=col, lw=2, label=f'{res["name"]} - ABM mean')
    ax1.fill_between(
        time, res["mean"] - res["std"], res["mean"] + res["std"], color=col, alpha=0.10
    )
    ax1.plot(
        time,
        res["N_det"],
        color=col,
        ls="--",
        lw=2,
        alpha=res["alpha"],
        label=f'{res["name"]} - Analytic',
    )
    for amt, dose_t in res["schedule"]:
        ax1.axvline(dose_t, color=col, ls=":", alpha=res["alpha"])

ax1.set_ylabel("Cells")
ax1.set_title("Tumour population: ABM vs Analytic (drug schedules)")
ax1.grid(True, alpha=res["alpha"])
ax1.legend(ncol=2, fontsize=9)

# PK comparison (deterministic)
for i, res in enumerate(results):
    col = scenario_palette[i % len(scenario_palette)]
    ax2.plot(
        time,
        res["conc_det"],
        color=col,
        lw=2,
        label=f'{res["name"]} (α={res["alpha"]})',
    )

ax2.set_xlabel("Time")
ax2.set_ylabel("Drug concentration")
ax2.set_title("PK profiles (drug schedules)")
ax2.grid(True, alpha=res["alpha"])
ax2.legend(ncol=2, fontsize=9)

plt.tight_layout()
plt.show()

print("\nSchedules compared:")
for res in results:
    print(f'- {res["name"]}: doses={res["schedule"]}, alpha={res["alpha"]}')
    print(f'  Final ABM: {res["mean"][-1]:.0f}, Analytic: {res["N_det"][-1]:.0f}')
