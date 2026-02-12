"""
LOGISTIC (resource-limited) ABM:
Compare normal logistic ABM vs resistant logistic ABM under identical schedules.

Top:
- Normal ABM total (dashed) vs Resistant ABM total (solid), mean ± std

Bottom:
- Resistant ABM composition (Sensitive vs Resistant) for each schedule

Side panel:
- key parameters + resistance metrics including t50 (first time R>S)
"""

import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import json
from pathlib import Path

from utils_logistic import (
    make_fragility_test_scenarios,
    process_scenarios_config,
    process_config,
    run_abm_logistic,
    run_abm_logistic_resistant,
)

# ---------------- Load Config ---------------- #
CONFIG_PATH = Path(__file__).parent / "config_logistic.json"
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

do_fragility_test = True
seed = 42

# ---------------- Build scenarios ---------------- #
if do_fragility_test:
    print("Running fragility test scenarios for logistic ABM...")

    x_bar = 40
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
else:
    print("Processing scenarios from config file...")
    scenarios = process_scenarios_config(config)

    # avoid NameError in side panel
    x_bar = None
    n_doses_per_cycle = None
    sigma = None
    n_cycles = None
    cycle_length = None
    alpha_val = None

# ---------------- Helper: t50 ---------------- #
def time_to_resistance_dominance(time, mean_SR):
    """
    t50 = first time where R(t) > S(t) using MEAN trajectory.
    Returns None if dominance never occurs.
    """
    if mean_SR.ndim != 2 or mean_SR.shape[1] != 2:
        return None
    S = mean_SR[:, 0]
    R = mean_SR[:, 1]
    idx = np.where(R > S)[0]
    if len(idx) == 0:
        return None
    return float(time[idx[0]])

# ---------------- Shared params (for panel) ---------------- #
initial_cells, birth_rate, death_rate, dt, steps, n_runs, hill_params = process_config(config)

# ---------------- Run normal + resistant ABMs ---------------- #
print("\nRunning NORMAL logistic ABM...")
abm_normal = run_abm_logistic(config, scenarios, seed=seed, compute_fragility=False)

print("Running RESISTANT logistic ABM...")
abm_resist = run_abm_logistic_resistant(config, scenarios, seed=seed, compute_fragility=False)

# ---------------- Pack results (robust time axis) ---------------- #
results = []
for i, sc in enumerate(scenarios):
    normal_mean = abm_normal["mean_trajectories"][i]  # (T,)
    normal_std  = abm_normal["std_trajectories"][i]   # (T,)

    resist_mean = abm_resist["mean_trajectories"][i]  # (T,2) -> (S,R)
    resist_std  = abm_resist["std_trajectories"][i]   # (T,2)

    if resist_mean.ndim != 2 or resist_mean.shape[1] != 2:
        raise ValueError(f"Expected resistant mean shape (T,2), got {resist_mean.shape}")

    # enforce same T (in case your normal/resistant use steps vs steps+1)
    T = min(len(normal_mean), len(resist_mean))
    normal_mean = normal_mean[:T]
    normal_std  = normal_std[:T]
    resist_mean = resist_mean[:T, :]
    resist_std  = resist_std[:T, :]

    time = np.arange(T) * dt

    resist_total_mean = resist_mean.sum(axis=1)
    # conservative envelope for total std (ok for plots)
    resist_total_std = resist_std.sum(axis=1) if resist_std.ndim == 2 else np.zeros_like(resist_total_mean)

    results.append(
        {
            "name": sc["name"],
            "schedule": sc["schedule"],
            "alpha": sc.get("alpha", None),
            "time": time,
            "normal_mean": normal_mean,
            "normal_std": normal_std,
            "resist_mean": resist_mean,              # (S,R)
            "resist_std": resist_std,
            "resist_total_mean": resist_total_mean,
            "resist_total_std": resist_total_std,
        }
    )

# ---------------- Plotting ---------------- #
scenario_palette = ["tab:orange", "tab:green", "tab:red", "tab:purple"]

n_sc = len(results)
fig = plt.figure(figsize=(14, 10))
gs = gridspec.GridSpec(2, n_sc, figure=fig, height_ratios=[1.05, 1.0])

ax_top = fig.add_subplot(gs[0, :])
ax_bottom = [fig.add_subplot(gs[1, i]) for i in range(n_sc)]

# ---- Top: Normal vs Resistant totals ----
for i, res in enumerate(results):
    col = scenario_palette[i % len(scenario_palette)]
    t = res["time"]
    
    # Resistant total (solid)
    ax_top.plot(t, res["resist_total_mean"], color=col, lw=2,
                label=f'{res["name"]} — Resistant ABM')
    ax_top.fill_between(t,
                        res["resist_total_mean"] - res["resist_total_std"],
                        res["resist_total_mean"] + res["resist_total_std"],
                        color=col, alpha=0.06)

    # Normal (dashed)
    ax_top.plot(t, res["normal_mean"], color=col, lw=2, ls="--", alpha=0.80,
                label=f'{res["name"]} — Normal ABM')
    ax_top.fill_between(t,
                        res["normal_mean"] - res["normal_std"],
                        res["normal_mean"] + res["normal_std"],
                        color=col, alpha=0.10)

    # Dose times
    for amt, dose_t in res["schedule"]:
        ax_top.axvline(dose_t, color=col, ls=":", alpha=0.25)

ax_top.set_ylabel("Cells")
ax_top.set_title("Tumour population: Normal ABM vs Resistant ABM")
ax_top.grid(alpha=0.3)
ax_top.legend(ncol=2, fontsize=9)

# ---- Bottom: Resistant composition ----
for i, res in enumerate(results):
    t = res["time"]
    S = res["resist_mean"][:, 0]
    R = res["resist_mean"][:, 1]

    ax_bottom[i].stackplot(
        t,
        S,
        R,
        labels=["Sensitive", "Resistant"],
        colors=["tab:blue", "tab:red"],
        alpha=0.80,
    )
    ax_bottom[i].set_title(f'{res["name"]}: Cell composition')
    ax_bottom[i].grid(alpha=0.3)
    ax_bottom[i].legend(loc="upper right", fontsize=9)
    ax_bottom[i].set_xlabel("Time")
    if i == 0:
        ax_bottom[i].set_ylabel("Cells")

# ---------------- Side panel: parameters + resistance summary + t50 ---------------- #
panel_lines = [
    "Simulation parameters",
    "----------------------",
    f"initial_cells = {initial_cells}",
    f"birth_rate    = {birth_rate}",
    f"death_rate    = {death_rate}",
    f"dt            = {dt}",
    f"steps         = {steps}",
    f"n_runs        = {n_runs}",
    f"p_mutation    = {config['simulation'].get('p_mutation', 'N/A')}",
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
    panel_lines += [
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

panel_lines += ["", "Resistance summary", "-----------------"]
for res in results:
    S_end = float(res["resist_mean"][-1, 0])
    R_end = float(res["resist_mean"][-1, 1])
    frac_end = R_end / (S_end + R_end) if (S_end + R_end) > 0 else 0.0
    t50 = time_to_resistance_dominance(res["time"], res["resist_mean"])

    panel_lines += [
        f"{res['name']}:",
        f"  final resistant fraction = {frac_end:.3f}",
        f"  final resistant cells    = {R_end:.0f}",
        f"  final sensitive cells    = {S_end:.0f}",
        f"  t50 (R>S)                = {t50:.2f}" if t50 is not None else "  t50 (R>S)                = n/a",
    ]

panel_text = "\n".join(panel_lines)

plt.tight_layout(rect=(0, 0, 0.82, 1.0))
fig.text(
    0.84,
    0.98,
    panel_text,
    va="top",
    ha="left",
    fontsize=8,
    family="monospace",
    bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.8", alpha=0.95),
)

outpath = Path(__file__).parent / "logistic_normal_vs_resistant.png"
plt.savefig(outpath, dpi=300)
plt.show()

print(f"\nSaved: {outpath}")

print("\nSchedules compared:")
for res in results:
    t50 = time_to_resistance_dominance(res["time"], res["resist_mean"])
    print(f"- {res['name']}: alpha={res['alpha']}, doses={len(res['schedule'])}")
    print(f"  Final normal ABM:    {res['normal_mean'][-1]:.0f}")
    print(f"  Final resistant ABM: {res['resist_total_mean'][-1]:.0f}")
    print(f"  t50 (R>S):           {t50:.2f}" if t50 is not None else "  t50 (R>S):           not reached")