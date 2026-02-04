"""
TGI-based fragility sweep (resource-dependent / logistic ABM)

We use time-integrated tumour burden (TGI) instead of endpoint size:
    TGI = ∫_0^T N(t) dt  (approximated numerically on the simulation grid)

Define "transient fragility" using TGI:
    F_TGI = (TGI_odd - TGI_even) / TGI0
where TGI0 is the no-treatment baseline over the same horizon.

Outputs:
- Plot of F_TGI vs mean dose x_bar (mean ± 1 SD across stochastic runs)
"""

import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt

from utils_logistic import (
    run_abm_logistic,
    run_abm_logistic_resistant,
    make_fragility_test_scenarios,
    process_config,
    logistic_fit,
)

# ---------------- Load Config ---------------- #
CONFIG_PATH = Path(__file__).parent / "config_logistic.json"
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

# ---------------- User sweep settings ---------------- #
x_bar_values = np.arange(10, 100, 5)

n_doses_per_cycle = 2
n_cycles = 4
cycle_length = 12
alpha_val = 1.0
target_frac_K = 0.85              # initialise near carrying capacity (adjust if desired)

base_seed = 42

# ---------------- Helpers ---------------- #
def set_horizon_steps(cfg, t_end):
    dt_local = cfg["simulation"]["dt"]
    cfg["simulation"]["steps"] = int(t_end / dt_local) + 1

def compute_tgi(trajs, dt):
    """
    trajs: array (n_runs, steps) of tumour populations N(t)
    Returns: array (n_runs,) of TGI values
    """
    # trapezoid rule over time grid
    return np.trapezoid(trajs, dx=dt, axis=1)

def as_total_2d(traj_obj):
    """
    Convert a trajectory object into a (n_runs, steps) array for TOTAL tumour N(t).
    Works for:
      - already-2D arrays
      - dict outputs from resistant ABM, e.g. {"total": arr, "S": arr, "R": arr}, etc.
    """
    # Case 1: already array-like
    if not isinstance(traj_obj, dict):
        arr = np.asarray(traj_obj)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D trajectories, got shape {arr.shape}")
        return arr

    # Case 2: dict-like (resistant ABM)
    keys = set(traj_obj.keys())

    # Prefer explicit total if present
    for k in ("total", "N", "tumour", "tumor", "population", "pop"):
        if k in keys:
            arr = np.asarray(traj_obj[k])
            if arr.ndim == 1:
                arr = arr[None, :]
            if arr.ndim != 2:
                raise ValueError(f"Key '{k}' has non-2D shape {arr.shape}")
            return arr

    # Otherwise try sum of sensitive + resistant
    # (common key conventions)
    candidates = [
        ("S", "R"),
        ("sensitive", "resistant"),
        ("sens", "res"),
    ]
    for ks, kr in candidates:
        if ks in keys and kr in keys:
            S = np.asarray(traj_obj[ks])
            R = np.asarray(traj_obj[kr])
            if S.ndim == 1: S = S[None, :]
            if R.ndim == 1: R = R[None, :]
            if S.shape != R.shape:
                raise ValueError(f"S and R shapes differ: {S.shape} vs {R.shape}")
            return S + R

    raise KeyError(f"Could not find total trajectories in dict keys: {sorted(keys)}")

# ============================================================
# 1) Estimate K from a LONG no-drug run (for setting initial N0)
# ============================================================
t_end_baseline = max(10 * cycle_length, config["simulation"]["steps"] * config["simulation"]["dt"])
set_horizon_steps(config, t_end_baseline)

initial_cells, birth_rate, death_rate, dt, steps, n_runs, hill_params = process_config(config)
time_baseline = np.arange(steps) * dt

baseline_scenario = [{"name": "No Drug", "schedule": [], "alpha": 0.0}]

original_n_runs = config["simulation"].get("n_runs", 1)
config["simulation"]["n_runs"] = 2

baseline_out = run_abm_logistic(config, baseline_scenario, seed=base_seed)

config["simulation"]["n_runs"] = original_n_runs

mean_no_drug = baseline_out["mean_trajectories"][0]
K0, r0, N0_0, baseline_fit = logistic_fit(time_baseline, mean_no_drug)

if K0 is None:
    raise RuntimeError("Logistic fit failed — cannot estimate carrying capacity K.")

# Set initial condition near K (based on fitted K)
config["simulation"]["initial_cells"] = int(target_frac_K * K0)

# ============================================================
# 2) Set horizon for the ACTUAL sweep (important!)
# ============================================================
t_end = n_cycles * cycle_length
set_horizon_steps(config, t_end)

initial_cells, birth_rate, death_rate, dt, steps, n_runs, hill_params = process_config(config)
time = np.arange(steps) * dt

# ============================================================
# 3) Compute TGI0 baseline for normalization (same horizon!)
# ============================================================
baseline_out2 = run_abm_logistic_resistant(config, baseline_scenario, seed=base_seed)
baseline_trajs = as_total_2d(baseline_out2["all_trajectories"][0])
TGI0_per_run = compute_tgi(baseline_trajs, dt)
TGI0_mean = float(TGI0_per_run.mean())

# ============================================================
# 4) Sweep x_bar and compute F_TGI
# ============================================================
F_TGI_mean = []
F_TGI_std = []

# optional: also store raw TGIs if you want to plot them later
TGI_even_mean, TGI_odd_mean = [], []

for j, x_bar in enumerate(x_bar_values):
    print(f"Computing TGI fragility for x_bar = {x_bar}...")

    total_dose_per_cycle = x_bar * n_doses_per_cycle

    # Ensure no negative doses: need sigma <= mean dose (= x_bar)
    sigma = x_bar

    scenarios = make_fragility_test_scenarios(
        total_dose=total_dose_per_cycle,
        n_doses=n_doses_per_cycle,
        n_cycles=n_cycles,
        sigma=sigma,
        cycle_length=cycle_length,
        alpha=alpha_val,
    )

    out = run_abm_logistic_resistant(config, scenarios, seed=base_seed + j)

    even_trajs = as_total_2d(out["all_trajectories"][0])
    odd_trajs  = as_total_2d(out["all_trajectories"][1])

    TGI_even = compute_tgi(even_trajs, dt)
    TGI_odd  = compute_tgi(odd_trajs, dt)

    # TGI-fragility per run
    F_TGI = (TGI_odd - TGI_even) / TGI0_mean

    F_TGI_mean.append(float(F_TGI.mean()))
    F_TGI_std.append(float(F_TGI.std(ddof=1)) if F_TGI.size > 1 else 0.0)

    TGI_even_mean.append(float(TGI_even.mean()))
    TGI_odd_mean.append(float(TGI_odd.mean()))

F_TGI_mean = np.array(F_TGI_mean, dtype=float)
F_TGI_std  = np.array(F_TGI_std, dtype=float)

# ============================================================
# 5) Plot
# ============================================================
fig, ax = plt.subplots(figsize=(12, 7))

ax.plot(x_bar_values, F_TGI_mean, lw=2, label="TGI fragility mean", color='blue')
ax.fill_between(
    x_bar_values,
    F_TGI_mean - F_TGI_std,
    F_TGI_mean + F_TGI_std,
    alpha=0.15,
    label="± SD",
    color='blue',
)
ax.axhline(0, ls="--", lw=1)

ax.set_title("TGI-based fragility vs mean dose (resource-dependent ABM)")
ax.set_xlabel(r"Mean dose $\bar{x}$")
ax.set_ylabel("TGI fragility")
ax.grid(True, alpha=0.3)
ax.legend()

# ---------------- Side parameter panel ---------------- #
panel_lines = [
    "Fragility parameters",
    "-------------------------------",
    f"alpha         = {alpha_val}",
    f"n_cycles      = {n_cycles}",
    f"cycle_length  = {cycle_length}",
    f"doses/cycle   = {n_doses_per_cycle}",
    f"sigma         = x_bar",
    f"horizon T     = {t_end}",
    "",
    "Model parameters",
    "------------",
    f"birth_rate     = {birth_rate}",
    f"death_rate     = {death_rate}",
    f"dt, steps      = {dt}, {steps}",
    f"n_runs         = {n_runs}",
    "",
    "Hill parameters",
    "----------",
    f"E0 = {hill_params['E0']}",
    f"E1 = {hill_params['E1']}",
    f"C  = {hill_params['C']}",
    f"n  = {hill_params['n']}",
]
panel_text = "\n".join(panel_lines)

plt.tight_layout(rect=(0, 0, 0.82, 1.0))
fig.text(
    0.84, 0.95, panel_text,
    va="top", ha="left",
    fontsize=8, family="monospace",
    bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.8", alpha=0.95),
)

outpath = Path(__file__).parent / "tgi_fragility_vs_xbar_resistant_multiple_cycles.png"
plt.savefig(outpath, dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved: {outpath}")