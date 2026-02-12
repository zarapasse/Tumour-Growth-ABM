"""
Compare normal exponential ABM vs resistant exponential ABM under the same schedules.

Top (A):
- Resistant ABM total (solid) vs Normal ABM total (dashed), mean ± std

Middle row:
- (B) Even schedule composition (Sensitive vs Resistant)
- (C) Odd schedule composition (Sensitive vs Resistant)

Bottom (D):
- PK profiles for both schedules

Right:
- Parameter + resistance summary side panel (never overlaps plots)
"""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

from utils import (
    make_fragility_test_scenarios,
    process_config,
    run_abm,
    run_abm_resistance,
    pk_concentration_series,
)

# ---------------------- Load config ---------------------- #
CONFIG_PATH = Path(__file__).parent / "config.json"
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

seed = 42
print("Running fragility test scenarios...")

# ---------------------- Scenario parameters ---------------------- #
x_bar = 30
n_doses_per_cycle = 2
total_dose_per_cycle = x_bar * n_doses_per_cycle
sigma = total_dose_per_cycle / 2
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
    p_mutation,
    initial_resistant_fraction,
) = process_config(config)

time = np.arange(steps + 1) * dt

# ---------------------- Run ABMs ---------------------- #
print("\nRunning normal ABM...")
abm_normal = run_abm(config, scenarios, seed=seed, compute_fragility=False)
time = abm_normal["time"]

print("Running resistant ABM...")
abm_resist = run_abm_resistance(
    config, scenarios, seed=seed, compute_fragility=False, enable_resistance=True
)

# sanity: ensure same time grid #! Delete once works
if not np.allclose(time, abm_resist["time"]):
    raise ValueError(
        "Normal and resistant ABMs returned different time grids. Check dt/steps consistency."
    )

# ---------------------- Results ---------------------- #
results = []
for i, sc in enumerate(scenarios):
    normal_mean = abm_normal["mean_trajectories"][i]  # (steps+1,)
    normal_std = abm_normal["std_trajectories"][i]  # (steps+1,)

    resist_total_mean = abm_resist["mean_total"][i]  # (steps+1,)
    resist_total_std = abm_resist["std_total"][i]  # (steps+1,)

    sens_mean = abm_resist["mean_sensitive"][i]  # (steps+1,)
    res_mean = abm_resist["mean_resistant"][i]  # (steps+1,)

    sens_std = abm_resist["std_sensitive"][i]
    res_std = abm_resist["std_resistant"][i]

    results.append(
        {
            "name": sc["name"],
            "schedule": sc["schedule"],
            "alpha": sc.get("alpha", None),
            "normal_mean": normal_mean,
            "normal_std": normal_std,
            "resist_total_mean": resist_total_mean,
            "resist_total_std": resist_total_std,
            "sens_mean": sens_mean,
            "res_mean": res_mean,
            "sens_std": sens_std,
            "res_std": res_std,
        }
    )


# ---------------------- Helpers ---------------------- #
def time_to_resistance_dominance(time, sens, res):
    """First time t such that R(t) > S(t) in the mean trajectories."""
    idx = np.where(res > sens)[0]
    return float(time[idx[0]]) if len(idx) else None


# ---------------------- Resistance summary ---------------------- #
res_frac_lines = ["", "Resistance summary", "---------------------------"]
for res in results:
    final_s = float(res["sens_mean"][-1])
    final_r = float(res["res_mean"][-1])
    final_tot = final_s + final_r
    final_frac = (final_r / final_tot) if final_tot > 0 else 0.0
    t50 = time_to_resistance_dominance(time, res["sens_mean"], res["res_mean"])

    res_frac_lines.append(f"{res['name']}:")
    res_frac_lines.append(f"  Resistance Fraction = {final_frac:.3f}")
    res_frac_lines.append(f"  Resistant Cells     = {final_r:.0f}")
    res_frac_lines.append(f"  Sensitive Cells     = {final_s:.0f}")
    res_frac_lines.append(
        f"  t50 (R>S)           = {t50:.2f}"
        if t50 is not None
        else "  t50 (R>S)           = n/a"
    )
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

# (B) even composition
sens_even = results[0]["sens_mean"]
res_even = results[0]["res_mean"]
axB.stackplot(
    time,
    sens_even,
    res_even,
    labels=["Sensitive", "Resistant"],
    colors=["tab:blue", "tab:red"],
    alpha=0.75,
)
axB.set_title(r"$\mathbf{(B)}$  Even schedule: composition", loc="left", fontsize=11)
axB.set_xlabel("Time (days)")
axB.set_ylabel("Cells")
axB.grid(True, alpha=0.3)
axB.legend(fontsize=9)

# (C) uneven composition
sens_odd = results[1]["sens_mean"]
res_odd = results[1]["res_mean"]
axC.stackplot(
    time,
    sens_odd,
    res_odd,
    labels=["Sensitive", "Resistant"],
    colors=["tab:blue", "tab:red"],
    alpha=0.75,
)
axC.set_title(r"$\mathbf{(C)}$  Uneven schedule: composition", loc="left", fontsize=11)
axC.set_xlabel("Time (days)")
axC.grid(True, alpha=0.3)
axC.legend(fontsize=9)

# (D) PK profiles
for i, sc in enumerate(scenarios):
    col = scenario_palette[i % len(scenario_palette)]
    conc = pk_concentration_series(time, sc["schedule"], float(sc["alpha"]))
    axD.plot(time, conc, color=col, lw=2, label=sc["name"])

axD.set_title(r"$\mathbf{(D)}$  PK profiles", loc="left", fontsize=12)
axD.set_xlabel("Time (days)")
axD.set_ylabel("Drug concentration")
axD.grid(True, alpha=0.3)
axD.legend(fontsize=9, ncol=2)

# ---------------------- Right panel ---------------------- #
param_lines = [
    "Simulation parameters",
    "----------------------",
    f"initial_cells = {initial_cells}",
    f"birth_rate    = {birth_rate}",
    f"death_rate    = {death_rate}",
    f"dt            = {dt}",
    f"steps         = {steps}",
    f"n_runs        = {n_runs}",
    f"p_mutation    = {p_mutation}",
    f"init_res_frac = {initial_resistant_fraction}",
    "",
    "Hill parameters",
    "--------------",
    f"K_kill = {hill_params.K_kill}",
    f"C      = {hill_params.C}",
    f"n      = {hill_params.n}",
    "",
    "Dosing parameters",
    "-----------------",
    f"x_bar            = {x_bar}",
    f"doses/cycle      = {n_doses_per_cycle}",
    f"sigma            = {sigma}",
    f"n_cycles         = {n_cycles}",
    f"cycle_length     = {cycle_length}",
    f"alpha            = {alpha_val}",
]
param_lines += res_frac_lines

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
    Path(__file__).parent / "Graphs/Resistance/exponential_normal_vs_resistant.png"
)
plt.savefig(outpath, dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved plot: {outpath}")

# ---------------------- Print summary ---------------------- #
print("\nSchedules compared:")
for res in results:
    final_normal = float(res["normal_mean"][-1])
    final_total = float(res["resist_total_mean"][-1])
    final_s = float(res["sens_mean"][-1])
    final_r = float(res["res_mean"][-1])

    print(f'- {res["name"]}: doses={len(res["schedule"])} doses, alpha={res["alpha"]}')
    print(f"  Final normal ABM:    {final_normal:.0f}")
    print(
        f"  Final resistant ABM: {final_total:.0f} (S: {final_s:.0f}, R: {final_r:.0f})"
    )
