import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

from utils import (
    make_fragility_test_scenarios,
    process_config,
    analytic_population_with_pk,
    run_abm_resistance,
)

CONFIG_PATH = Path(__file__).parent / "config.json"
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

print("Running fragility test scenarios...")

# ---- scenario params ----
x_bar = 20
n_doses_per_cycle = 2
total_dose_per_cycle = x_bar * n_doses_per_cycle
sigma = total_dose_per_cycle / 2
n_cycles = 4
cycle_length = 12
alpha_val = 1.0

scenarios = make_fragility_test_scenarios(
    total_dose_per_cycle, n_doses_per_cycle, n_cycles, sigma, cycle_length, alpha_val
)

# ---- model params ----
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
    fitness_cost,
) = process_config(config)

# ---- ABM (resistant) ----
abm = run_abm_resistance(
    config, scenarios, seed=42, compute_fragility=True, enable_resistance=True
)
time = abm["time"]

# ---- Deterministic (no resistance) ----
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
    results.append({"N_det": N_det, "conc_det": conc_det})

# ---- Print fragility ----
frag = abm.get("fragility")
if frag is not None:
    print("\n--- Fragility Analysis ---")
    print(f"Mean Fragility: {frag['mean']:.4f}")
    print(f"Std  Fragility: {frag['std']:.4f}")
    print(f"Per-run Fragility: {frag['per_run']}")

# ========== PLOTTING (same layout as your old code) ==========
scenario_palette = ["tab:orange", "tab:green", "tab:red", "tab:purple"]

fig = plt.figure(figsize=(15, 10))
gs = gridspec.GridSpec(
    3,
    3,
    width_ratios=[1.0, 1.0, 0.55],  # <- panel column
    height_ratios=[1.2, 1.0, 0.8],  # A, (B/C), D
    hspace=0.40,
    wspace=0.30,
)

axA = fig.add_subplot(gs[0, 0:2])  # (A) spans 2 plot columns
axB = fig.add_subplot(gs[1, 0])  # (B) bottom-left
axC = fig.add_subplot(gs[1, 1])  # (C) bottom-right
axD = fig.add_subplot(gs[2, 0:2])  # (D) spans 2 plot columns

# Panel occupies the full right column
axP = fig.add_subplot(gs[:, 2])
axP.axis("off")  # hide axis lines/ticks

# ---------------- (A) Total population ----------------
for i, sc in enumerate(scenarios):
    col = scenario_palette[i % len(scenario_palette)]

    axA.plot(
        time,
        abm["mean_total"][i],
        color=col,
        lw=2,
        label=f"{sc['name']} — ABM",
    )
    axA.fill_between(
        time,
        abm["mean_total"][i] - abm["std_total"][i],
        abm["mean_total"][i] + abm["std_total"][i],
        color=col,
        alpha=0.12,
    )

    axA.plot(
        time,
        results[i]["N_det"],
        color=col,
        ls="--",
        lw=2,
        alpha=0.35,
        label=f"{sc['name']} — Deterministic",
    )

    for _, dose_t in sc["schedule"]:
        axA.axvline(dose_t, color=col, ls=":", alpha=0.25)

axA.set_title(r"$\mathbf{(A)}$  Tumour population", loc="left", fontsize=12)
axA.set_ylabel("Cells")
axA.grid(True, alpha=0.3)
axA.legend(ncol=2, fontsize=9)

# ---------------- (B) Composition: Even ----------------
sens_even = abm["mean_sensitive"][0]
res_even = abm["mean_resistant"][0]

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

# ---------------- (C) Composition: Uneven ----------------
sens_odd = abm["mean_sensitive"][1]
res_odd = abm["mean_resistant"][1]

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

# ---------------- (D) PK profiles ----------------
for i, sc in enumerate(scenarios):
    col = scenario_palette[i % len(scenario_palette)]
    axD.plot(time, results[i]["conc_det"], color=col, lw=2, label=sc["name"])

axD.set_title(r"$\mathbf{(D)}$  PK profiles", loc="left", fontsize=12)
axD.set_xlabel("Time (days)")
axD.set_ylabel("Drug concentration")
axD.grid(True, alpha=0.3)
axD.legend(fontsize=9, ncol=2)

# ---- side parameter panel (like your old script) ----
sim = config["simulation"]
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
# ---- Right parameter panel ----
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

outpath = Path(__file__).parent / "Graphs/Resistance/exponential_resistance_trajectory.png"
plt.savefig(outpath, dpi=300, bbox_inches="tight")
#plt.show()
plt.close()

print(f"Saved plot: {outpath}")
