"""
Compare normal exponential ABM vs resistant exponential ABM under the same schedules.

Top:
- Normal ABM total (solid) vs Resistant ABM total (dashed), mean ± std

Bottom:
- Resistant ABM composition (Sensitive vs Resistant), one subplot per schedule

Figure includes parameter + resistance summary side panel.
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
    run_abm,
    run_abm_for_resistant,
)

# ---------------------- Load config ---------------------- #
CONFIG_PATH = Path(__file__).parent / "config.json"
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

do_fragility_test = True
seed = 42

# ---------------------- Scenarios ---------------------- #
if do_fragility_test:
    print("Running fragility test scenarios...")

    # Hardcoded values (edit freely)
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
else:
    print("Processing scenarios from config file...")
    scenarios = process_scenarios_config(config)

    # For parameter panel (avoid NameError)
    x_bar = None
    n_doses_per_cycle = None
    sigma = None
    n_cycles = None
    cycle_length = None
    alpha_val = None

# ---------------------- Shared sim params ---------------------- #
initial_cells, birth_rate, death_rate, dt, steps, n_runs, hill_params = process_config(config)
time = np.arange(steps) * dt

# ---------------------- Run ABMs ---------------------- #
print("\nRunning normal ABM...")
abm_normal = run_abm(config, scenarios, seed=seed, compute_fragility=False)

print("Running resistant ABM...")
abm_resist = run_abm_for_resistant(config, scenarios, seed=seed, compute_fragility=False)

# ---------------------- Pack results ---------------------- #
results = []
for i, sc in enumerate(scenarios):
    normal_mean = abm_normal["mean_trajectories"][i]      # (steps,)
    normal_std  = abm_normal["std_trajectories"][i]       # (steps,)

    resist_mean = abm_resist["mean_trajectories"][i]      # (steps,2)
    resist_std  = abm_resist["std_trajectories"][i]       # (steps,2)

    if resist_mean.ndim != 2 or resist_mean.shape[1] != 2:
        raise ValueError(f"Expected resistant mean shape (steps,2), got {resist_mean.shape}")

    # Totals
    resist_total_mean = resist_mean.sum(axis=1)
    # Conservative total std envelope (std_S + std_R). Fine for plotting.
    resist_total_std  = resist_std.sum(axis=1) if resist_std.ndim == 2 else np.zeros_like(resist_total_mean)

    results.append(
        {
            "name": sc["name"],
            "schedule": sc["schedule"],
            "alpha": sc.get("alpha", None),
            "normal_mean": normal_mean,
            "normal_std": normal_std,
            "resist_mean": resist_mean,              # (S,R)
            "resist_std": resist_std,                # (S,R)
            "resist_total_mean": resist_total_mean,
            "resist_total_std": resist_total_std,
        }
    )

# ---------------------- Plotting ---------------------- #
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

n_sc = len(results)

fig = plt.figure(figsize=(14, 10))

# Layout: top spans all columns, bottom has one plot per schedule
gs = gridspec.GridSpec(2, n_sc, figure=fig, height_ratios=[1, 1])
ax1 = fig.add_subplot(gs[0, :])
ax2_list = [fig.add_subplot(gs[1, i]) for i in range(n_sc)]

# ------------------ 1) Top: Normal vs Resistant totals ------------------ #
for i, res in enumerate(results):
    col = scenario_palette[i % len(scenario_palette)]

    # Resistant ABM total 
    ax1.plot(time, res["resist_total_mean"], color=col, lw=2,
             label=f'{res["name"]} — Resistant ABM')
    ax1.fill_between(time,
                     res["resist_total_mean"] - res["resist_total_std"],
                     res["resist_total_mean"] + res["resist_total_std"],
                     color=col, alpha=0.06)


    # Normal ABM 
    ax1.plot(time, res["normal_mean"], color=col, lw=2, ls="--", alpha =0.8,
             label=f'{res["name"]} — Normal ABM')
    ax1.fill_between(time,
                     res["normal_mean"] - res["normal_std"],
                     res["normal_mean"] + res["normal_std"],
                     color=col, alpha=0.10)
    # Dose times
    for amt, dose_t in res["schedule"]:
        ax1.axvline(dose_t, color=col, ls=":", alpha=0.25)

ax1.set_ylabel("Population")
ax1.set_title("Tumour population: Normal ABM vs Resistant ABM")
ax1.grid(True, alpha=0.3)
ax1.legend(ncol=2, fontsize=9)

# ------------------ 2) Bottom: Resistant composition ------------------ #
for i, res in enumerate(results):
    sens_mean = res["resist_mean"][:, 0]
    res_mean  = res["resist_mean"][:, 1]

    ax2_list[i].stackplot(
        time,
        sens_mean,
        res_mean,
        labels=["Sensitive", "Resistant"],
        colors=["tab:blue", "tab:red"],
        alpha=0.75,
    )
    ax2_list[i].set_title(f'{res["name"]}: Cell Composition')
    ax2_list[i].grid(True, alpha=0.3)
    ax2_list[i].legend(loc="upper right", fontsize=9)

    if i == 0:
        ax2_list[i].set_ylabel("Cells")
    ax2_list[i].set_xlabel("Time")
    
    
def time_to_resistance_dominance(time, mean_SR):
    """
    Return t50 = first time where R(t) > S(t), using the mean trajectory.
    If dominance never occurs, return None.
    """
    if mean_SR.ndim != 2 or mean_SR.shape[1] != 2:
        return None

    S = mean_SR[:, 0]
    R = mean_SR[:, 1]

    idx = np.where(R > S)[0]
    if len(idx) == 0:
        return None

    return float(time[idx[0]])

# ---------------------- Resistance summary ---------------------- #
res_frac_lines = ["", "Resistance summary", "---------------------------"]
for res in results:
    final_sens = float(res["resist_mean"][-1, 0])
    final_res  = float(res["resist_mean"][-1, 1])
    final_tot  = final_sens + final_res
    final_frac = (final_res / final_tot) if final_tot > 0 else 0.0

    t50 = time_to_resistance_dominance(time, res["resist_mean"])

    res_frac_lines.append(f"{res['name']}:")
    res_frac_lines.append(f"  Resistance Fraction = {final_frac:.3f}")
    res_frac_lines.append(f"  Resistant Cells     = {final_res:.0f}")
    res_frac_lines.append(f"  Sensitive Cells     = {final_sens:.0f}")
    res_frac_lines.append(f"  t50 (R>S)           = {t50:.2f}" if t50 is not None else "  t50 (R>S)           = n/a")

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
    f"initial_resistant_fraction = {config['simulation'].get('initial_resistant_fraction', 'N/A')}",
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
        "Dosing parameters",
        "-----------------",
        f"x_bar            = {x_bar}",
        f"doses per cycle  = {n_doses_per_cycle}",
        f"sigma            = {sigma}",
        f"n_cycles         = {n_cycles}",
        f"cycle_length     = {cycle_length}",
        f"alpha            = {alpha_val}",
    ]

param_lines += res_frac_lines
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

outpath = Path(__file__).parent / "exponential_normal_vs_resistant.png"
plt.savefig(outpath, dpi=300)
plt.show()

# ---------------------- Print summary ---------------------- #
print("\nSchedules compared:")
for res in results:
    final_normal = float(res["normal_mean"][-1])
    final_resist = float(res["resist_total_mean"][-1])
    final_sens   = float(res["resist_mean"][-1, 0])
    final_r      = float(res["resist_mean"][-1, 1])

    print(f'- {res["name"]}: doses={len(res["schedule"])} doses, alpha={res["alpha"]}')
    print(f"  Final normal ABM:    {final_normal:.0f}")
    print(f"  Final resistant ABM: {final_resist:.0f} (S: {final_sens:.0f}, R: {final_r:.0f})")