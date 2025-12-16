""" 
This script compares analytic fragility calculations with ABM results.
It generates a figure with two tables: one for analytic fragility (no resistance)
and one for ABM fragility (with/without resistance depending on function used).

Parameters are explicitly stated at the top of the figure for clarity.
"""

import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt

from utils import (
    make_fragility_test_scenarios,
    process_config,
    run_abm_for_resistant,
)

# ----------------- Load config ----------------- #
CONFIG_PATH = Path(__file__).parent / "config.json"
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

# Extract Hill parameters
n = config["hill_parameters"]["n"]
C = config["hill_parameters"]["C"]

# Fixed parameters
n_doses_per_cycle = 2
n_cycles = 4
cycle_length = 12
alpha = 1
T = cycle_length

# Process ABM config to get dt etc.
initial_cells, birth_rate, death_rate, dt, steps, n_runs, hill_params = process_config(config)

# Sweep x_bar values
x_bar_values = np.arange(10, 100, 5)

# ----------------- Analytic fragility function ----------------- #
def analytic_fragility(x_bar, sigma):
    k = np.exp(-alpha * n * T / 2)

    term_plus  = ((x_bar + sigma)**n * k + C**n) / ((x_bar + sigma)**n + C**n)
    term_minus = ((x_bar - sigma)**n * k + C**n) / ((x_bar - sigma)**n + C**n)
    term_even  = (x_bar**n         * k + C**n) / (x_bar**n + C**n)

    return np.log(term_plus) + np.log(term_minus) - 2*np.log(term_even)

# ----------------- Compute analytic fragility ----------------- #
analytic_table = []
for x_bar in x_bar_values:
    total_dose = x_bar * n_doses_per_cycle
    sigma = 0.5 * total_dose
    F = analytic_fragility(x_bar, sigma)
    analytic_table.append([f"{x_bar:.2f}", f"{F:.6f}"])

# ----------------- Compute ABM fragility ----------------- #
abm_table = []
for x_bar in x_bar_values:
    total_dose_per_cycle = x_bar * n_doses_per_cycle
    sigma = total_dose_per_cycle / 2

    scenarios = make_fragility_test_scenarios(
        total_dose_per_cycle,
        n_doses_per_cycle,
        n_cycles,
        sigma,
        cycle_length,
        alpha,
    )

    abm_results = run_abm_for_resistant(config, scenarios, seed=42, compute_fragility=True)
    frag = np.mean(np.array(abm_results["fragility"]["per_run"], dtype=float))

    abm_table.append([f"{x_bar:.2f}", f"{frag:.6f}"])

# ----------------- Build figure with two tables ----------------- #
fig, axes = plt.subplots(1, 2, figsize=(12, 11))
plt.subplots_adjust(top=0.75, wspace=0.15)

# Table headers
col_labels = ["x_bar", "Fragility"]

# ----- Parameter Key (TOP) ----- #
param_text = (
    "Parameters Used:\n"
    f"  Hill params: n={n}, C={C}\n"
    f"  α = {alpha}\n"
    f"  dt = {dt}\n"
    f"  Doses per cycle = {n_doses_per_cycle}\n"
    f"  Cycles = {n_cycles}\n"
    f"  Cycle length = {cycle_length}\n"
    f"  σ = total_dose/2\n"
    f"  n_runs (ABM) = {n_runs}"
)

fig.text(
    0.5, 0.93, param_text,
    va='top', ha='center',
    fontsize=9, family='monospace',
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.9)
)

# ----- Analytic Table ----- #
axes[0].axis('off')
tab1 = axes[0].table(
    cellText=analytic_table,
    colLabels=col_labels,
    cellLoc='center',
    loc='center'
)
tab1.auto_set_font_size(False)
tab1.set_fontsize(9)
tab1.scale(1.2, 1.4)
axes[0].set_title("Analytic Fragility (No Resistance)", fontsize=12, pad=10)

# ----- ABM Table ----- #
axes[1].axis('off')
tab2 = axes[1].table(
    cellText=abm_table,
    colLabels=col_labels,
    cellLoc='center',
    loc='center'
)
tab2.auto_set_font_size(False)
tab2.set_fontsize(9)
tab2.scale(1.2, 1.4)
axes[1].set_title("ABM Fragility (With Resistance)", fontsize=12, pad=10)

# ----- Save figure ----- #
output_path = Path(__file__).parent / "fragility_comparison_fitness0.8_cost.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"Saved analytic vs ABM fragility comparison to: {output_path}")

