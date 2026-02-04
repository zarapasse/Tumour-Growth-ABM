import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt

from utils_logistic import (
    run_abm_logistic,
    run_abm_logistic_resistant,
    make_fragility_test_scenarios,
    process_config,
)

# ---------------- Load Config ---------------- #
CONFIG_PATH = Path(__file__).parent / "config_logistic.json"
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

# ---------------- Sweep parameters ---------------- #
x_bar_values = np.arange(10, 100, 5)

n_doses_per_cycle = 2
n_cycles = 4
cycle_length = 12
alpha_val = 1.0

sigma_fixed = 7.0
target_frac_K = 0.05

base_seed = 42  # reproducibility

# ---------------- Helper: set steps from horizon ---------------- #
def set_horizon_steps(cfg, t_end):
    dt_local = cfg["simulation"]["dt"]
    cfg["simulation"]["steps"] = int(t_end / dt_local) + 1

# ================================================================
# 1) Estimate K from NO-DRUG logistic ABM (plateau, no fitting)
# ================================================================

# run long enough to reach steady state
dt0 = config["simulation"]["dt"]
t_end_baseline = max(30 * cycle_length, config["simulation"]["steps"] * dt0)
set_horizon_steps(config, t_end_baseline)

baseline_out = run_abm_logistic(
    config,
    [{"name": "No Drug", "schedule": [], "alpha": 0.0}],
    seed=base_seed,
)

mean_no_drug = baseline_out["mean_trajectories"][0]

# plateau estimate: last 20% (min 10 points)
tail_n = max(10, int(0.2 * len(mean_no_drug)))
tail = mean_no_drug[-tail_n:]

K0 = float(np.median(tail))
if not np.isfinite(K0) or K0 <= 0:
    raise RuntimeError("Plateau-based K estimate failed.")

# set initial condition as fraction of K
config["simulation"]["initial_cells"] = int(target_frac_K * K0)

print(
    f"[baseline] Estimated K ≈ {K0:.1f}, "
    f"initial_cells = {config['simulation']['initial_cells']} "
    f"({target_frac_K:.2f}K)"
)

# ================================================================
# 2) Set horizon for ACTUAL fragility sweep
# ================================================================

t_end = n_cycles * cycle_length
set_horizon_steps(config, t_end)

initial_cells, birth_rate, death_rate, dt, steps, n_runs, hill_params = process_config(config)
time = np.arange(steps) * dt

# ================================================================
# 3) Sweep fragility vs mean dose
# ================================================================

frag_mean = []
frag_std = []

for j, x_bar in enumerate(x_bar_values):
    print(f"Computing fragility for x_bar = {x_bar}...")

    total_dose_per_cycle = x_bar * n_doses_per_cycle

    # ensure no negative doses
    sigma = min(sigma_fixed, 0.95 * x_bar)

    scenarios = make_fragility_test_scenarios(
        total_dose=total_dose_per_cycle,
        n_doses=n_doses_per_cycle,
        n_cycles=n_cycles,
        sigma=sigma,
        cycle_length=cycle_length,
        alpha=alpha_val,
    )

    out = run_abm_logistic_resistant(
        config,
        scenarios,
        seed=base_seed + j,
        compute_fragility=True,
    )

    frag_per_run = out["fragility"]["per_run"] / float(initial_cells)

    frag_mean.append(frag_per_run.mean())
    frag_std.append(frag_per_run.std(ddof=1) if frag_per_run.size > 1 else 0.0)

frag_mean = np.asarray(frag_mean)
frag_std = np.asarray(frag_std)

# ================================================================
# 4) Plot
# ================================================================

fig, ax = plt.subplots(figsize=(12, 7))

ax.plot(x_bar_values, frag_mean, lw=2, label="ABM mean", color="blue")
ax.fill_between(
    x_bar_values,
    frag_mean - frag_std,
    frag_mean + frag_std,
    alpha=0.15,
    color="blue",
    label="ABM ± SD",
)

ax.axhline(0, ls="--", lw=1)
ax.set_title("Fragility vs mean dose (resistant logistic ABM)")
ax.set_xlabel(r"Mean dose $\bar{x}$")
ax.set_ylabel("Fragility")
ax.grid(True, alpha=0.3)
ax.legend()

# ---------------- Side parameter panel ---------------- #
panel_lines = [
    "Fragility sweep",
    "------------------------------",
    f"alpha         = {alpha_val}",
    f"n_cycles      = {n_cycles}",
    f"cycle_length  = {cycle_length}",
    f"doses/cycle   = {n_doses_per_cycle}",
    "",
    "Model params",
    "------------",
    f"birth_rate    = {birth_rate}",
    f"death_rate    = {death_rate}",
    f"dt, steps     = {dt}, {steps}",
    f"n_runs        = {n_runs}",
    f"p_mutation    = {config['simulation']['p_mutation']}",
    f"initial_resistant_fraction = {config['simulation'].get('initial_resistant_fraction', 'N/A')}",
    "",
    "Hill params",
    "----------",
    f"E0 = {hill_params['E0']}",
    f"E1 = {hill_params['E1']}",
    f"C  = {hill_params['C']}",
    f"n  = {hill_params['n']}",
]

panel_text = "\n".join(panel_lines)

plt.tight_layout(rect=(0, 0, 0.82, 1.0))
fig.text(
    0.84, 0.95,
    panel_text,
    va="top",
    ha="left",
    fontsize=8,
    family="monospace",
    bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.8", alpha=0.95),
)

outpath = Path(__file__).parent / "fragility_vs_xbar_resistant_logistic_multiple_cycles.png"
plt.savefig(outpath, dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved: {outpath}")