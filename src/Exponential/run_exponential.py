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
)

# Load parameters from JSON file
CONFIG_PATH = Path(__file__).parent / "config.json"
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

do_fragility_test = True

if do_fragility_test:
    print("Running fragility test scenarios...")

    #! Hardcoded values for dosing analysis
    x_bar = 20
    n_doses_per_cycle = 2
    total_dose_per_cycle = x_bar * n_doses_per_cycle
    sigma = total_dose_per_cycle / 2      # holiday example
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

initial_cells, birth_rate, death_rate, dt, steps, n_runs, hill_params = process_config(config)
time = np.arange(steps) * dt

if is_fragility:
    abm_results = run_abm(config, scenarios, seed=42, compute_fragility=True)

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

    fragility_info = abm_results["fragility"]
    print(f"\n--- Fragility Analysis ---")
    print(f"Mean Fragility: {fragility_info['mean']:.4f}")
    print(f"Per-run Fragility: {fragility_info['per_run']}")

else:
    for sc in scenarios:
        abm_out = run_abm(config, [sc], seed=42)

        all_runs = abm_out["all_trajectories"][0]  # shape (n_runs, steps)
        mean_abm = all_runs.mean(axis=0)
        std_abm = all_runs.std(axis=0)

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

# ========== PLOTTING ==========
scenario_palette = [
    "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown",
    "tab:pink", "tab:gray", "tab:olive", "tab:cyan",
]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 10), sharex=False)

# -------- (A) Tumour trajectory --------
for i, res in enumerate(results):
    col = scenario_palette[i % len(scenario_palette)]
    ax1.plot(time, res["mean"], color=col, lw=2, label=f'{res["name"]} — ABM')
    ax1.fill_between(
        time,
        res["mean"] - res["std"],
        res["mean"] + res["std"],
        color=col,
        alpha=0.10,
    )
    ax1.plot(
        time,
        res["N_det"],
        color=col,
        ls="--",
        lw=2,
        alpha=0.3,
        label=f'{res["name"]} — Deterministic',
    )
    for amt, dose_t in res["schedule"]:
        ax1.axvline(dose_t, color=col, ls=":", alpha=0.3)

# Panel label (left) + title (center)
ax1.set_title(r"$\mathbf{(A)}$", loc="left", fontsize=12, pad=6)
ax1.set_title("Tumour population: ABM vs Deterministic", loc="center", fontsize=13, pad=6)

ax1.set_xlabel("Time (days)")
ax1.set_ylabel("Tumour population (cells)")
ax1.grid(True, alpha=0.3)
ax1.legend(ncol=2, fontsize=9)

# -------- (B) PK profiles --------
for i, res in enumerate(results):
    col = scenario_palette[i % len(scenario_palette)]
    ax2.plot(
        time,
        res["conc_det"],
        color=col,
        lw=2,
        label=f'{res["name"]}',
    )

# Panel label (left) + title (center)
ax2.set_title(r"$\mathbf{(B)}$", loc="left", fontsize=12, pad=6)
ax2.set_title("PK profiles (drug schedules)", loc="center", fontsize=12, pad=6)

ax2.set_xlabel("Time (days)")
ax2.set_ylabel("Drug concentration (mg/L)")
ax2.grid(True, alpha=0.3)
ax2.legend(ncol=2, fontsize=9)

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
    "",
    "Hill parameters",
    "--------------",
    f"E0 = {hill_params['E0']}",
    f"E1 = {hill_params['E1']}",
    f"C  = {hill_params['C']}",
    f"n  = {hill_params['n']}",
]

if do_fragility_test:
    param_lines += [
        "",
        "Schedule parameters",
        "-------------------",
        f"x_bar        = {x_bar}",
        f"sigma        = {sigma}",
        f"n_cycles     = {n_cycles}",
        f"cycle_length = {cycle_length}",
        f"alpha        = {alpha_val}",
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
    bbox=dict(
        boxstyle="round",
        facecolor="white",
        edgecolor="0.8",
        alpha=0.95,
    ),
)

plt.savefig(Path(__file__).parent / "exponential_trajectory.png", dpi=300)
plt.show()

print("\nSchedules compared:")
for res in results:
    print(f'- {res["name"]}: doses={res["schedule"]}, alpha={res["alpha"]}')
    print(f'  Final ABM: {res["mean"][-1]:.0f}, Analytic: {res["N_det"][-1]:.0f}')