import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt

from utils_logistic import (
    run_abm_logistic,
    make_fragility_test_scenarios,
    process_config,
    logistic_fit_windowed,          # <-- NEW (windowed fit)
    continuum_logistic_with_pkpd,   # <-- deterministic comparator
)

# ============================================================
#                      CONFIG + SETTINGS
# ============================================================
CONFIG_PATH = Path(__file__).parent / "config_logistic.json"
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

# Sweep over mean dose per administration
x_bar_values = np.arange(10, 100, 5)

# Dosing protocol
n_doses_per_cycle = 2
n_cycles = 4
cycle_length = 12
alpha_val = 1.0

# "Maximal uneven" for n_doses=2: sigma = mean dose = x_bar
# (gives [2 x_bar, 0] per cycle when mean is x_bar)
use_maximal_uneven = True

# Baseline fit settings
fit_window_start = 10.0         # fit K,r to no-drug AFTER this time
n_baseline_cycles = 10          # run long baseline so K is well-estimated

# Initial condition for the fragility sweep:
# set the tumour near carrying capacity estimated from the baseline fit
target_frac_K = 0.05

# Reproducibility
base_seed = 42

# ============================================================
#                    HELPERS (time horizon)
# ============================================================
def set_horizon_steps(cfg, t_end):
    dt_local = cfg["simulation"]["dt"]
    cfg["simulation"]["steps"] = int(t_end / dt_local) + 1

# ============================================================
#      1) LONG NO-DRUG RUN TO ESTIMATE EMERGENT K (WINDOWED)
# ============================================================
t_end_baseline = n_baseline_cycles * cycle_length
set_horizon_steps(config, t_end_baseline)

initial_cells, birth_rate, death_rate, dt, steps, n_runs, hill_params = process_config(config)
time_baseline = np.arange(steps) * dt

baseline_scenario = [{"name": "No Drug", "schedule": [], "alpha": 0.0}]
baseline_out = run_abm_logistic(config, baseline_scenario, seed=base_seed)

mean_no_drug = baseline_out["mean_trajectories"][0]

# Windowed logistic fit (this is the “fit style you trust”)
K0, r0, N0_fit, baseline_fit_full = logistic_fit_windowed(
    time_baseline, mean_no_drug, t_start=fit_window_start
)

if K0 is None:
    raise RuntimeError("Windowed logistic fit failed — cannot estimate carrying capacity K.")

print(f"[baseline fit] window start = {fit_window_start}")
print(f"[baseline fit] K0 = {K0:.3f}, r0 = {r0:.4f}, N0_fit = {N0_fit:.3f}")

# Set IC near carrying capacity for the fragility sweep
config["simulation"]["initial_cells"] = max(1, int(target_frac_K * K0))

# ============================================================
#          2) SET HORIZON FOR THE FRAGILITY SWEEP
# ============================================================
t_end = n_cycles * cycle_length
set_horizon_steps(config, t_end)

initial_cells, birth_rate, death_rate, dt, steps, n_runs, hill_params = process_config(config)
time = np.arange(steps) * dt

# If you want the deterministic comparator to start at the same post-transient
# state used in the windowed fit:
i0 = int(round(fit_window_start / dt))
i0 = min(max(i0, 0), len(time) - 1)

# NOTE: the ABM starts at t=0 with `initial_cells` agents.
# The deterministic model can't simultaneously start at t=0 AND match the ABM
# at t=fit_window_start unless you explicitly re-initialise it.
# We'll do a clean comparator by matching at t=fit_window_start:
N0_det = float(initial_cells)

# ============================================================
#              3) SWEEP FRAGILITY (ABM + CONTINUUM)
# ============================================================
frag_mean = []
frag_std = []
frag_det = []

for j, x_bar in enumerate(x_bar_values):
    print(f"Computing fragility for x_bar = {x_bar}...")

    total_dose_per_cycle = x_bar * n_doses_per_cycle

    # For n_doses=2:
    # mean_dose = x_bar, sigma <= mean_dose to avoid negative doses.
    if use_maximal_uneven:
        sigma = x_bar  # maximal uneven => [2x_bar, 0]
    else:
        # e.g. half-mean uneven if you ever want it:
        sigma = 0.5 * x_bar

    scenarios = make_fragility_test_scenarios(
        total_dose=total_dose_per_cycle,
        n_doses=n_doses_per_cycle,
        n_cycles=n_cycles,
        sigma=sigma,
        cycle_length=cycle_length,
        alpha=alpha_val,
    )

    # ---------------- ABM fragility ----------------
    out = run_abm_logistic(
        config,
        scenarios,
        seed=base_seed + j,          # vary seed across sweep
        compute_fragility=True,
    )
    frag_per_run = out["fragility"]["per_run"] / float(initial_cells)

    frag_mean.append(frag_per_run.mean())
    frag_std.append(frag_per_run.std(ddof=1) if frag_per_run.size > 1 else 0.0)

    # ---------------- Continuum fragility ----------------
    # Simulate deterministic comparator for EVEN and ODD schedules.
    # Use K0,r0 from the no-drug windowed fit.
    N_even, _, _, _ = continuum_logistic_with_pkpd(
        time=time,
        dt=dt,
        N0=N0_det,
        K=K0,
        r=r0,
        schedule=scenarios[0]["schedule"],
        alpha=scenarios[0]["alpha"],
        hill_params=hill_params,
        lag_one_step=True,
    )

    N_odd, _, _, _ = continuum_logistic_with_pkpd(
        time=time,
        dt=dt,
        N0=N0_det,
        K=K0,
        r=r0,
        schedule=scenarios[1]["schedule"],
        alpha=scenarios[1]["alpha"],
        hill_params=hill_params,
        lag_one_step=True,
    )

    frag_det.append((N_odd[-1] - N_even[-1]) / float(initial_cells))

frag_mean = np.asarray(frag_mean, dtype=float)
frag_std  = np.asarray(frag_std, dtype=float)
frag_det  = np.asarray(frag_det, dtype=float)

# ============================================================
#                          4) PLOT
# ============================================================
fig, ax = plt.subplots(figsize=(12, 7))

ax.plot(x_bar_values, frag_mean, lw=2, label="ABM mean", color="blue")
ax.fill_between(
    x_bar_values,
    frag_mean - frag_std,
    frag_mean + frag_std,
    alpha=0.15,
    label="ABM ± SD",
    color="blue",
)

ax.plot(
    x_bar_values,
    frag_det,
    lw=2,
    ls="--",
    label="Logistic + PK/PD",
    color = "black"
)

ax.axhline(0, ls="--", lw=1)

ax.set_title("Fragility vs mean dose (resource-dependent ABM)")
ax.set_xlabel(r"Mean dose $\bar{x}$ (mg/L)")
ax.set_ylabel("Fragility")
ax.grid(True, alpha=0.3)
ax.legend()

# ---------------- Side parameter panel ----------------
panel_lines = [
    "Fragility sweep",
    "------------------------------------------",
    f"alpha            = {alpha_val}",
    f"n_cycles         = {n_cycles}",
    f"cycle_length     = {cycle_length}",
    f"doses/cycle      = {n_doses_per_cycle}",
    "Logistic fit (no drug)",
    "----------------------",
    f"K0 ≈ {K0:.1f}",
    f"r0 ≈ {r0:.4f}",
    "",
    "Model parameters",
    "---------------",
    f"birth_rate        = {birth_rate}",
    f"death_rate        = {death_rate}",
    f"dt, steps         = {dt}, {steps}",
    f"n_runs            = {n_runs}",
    "",
    "Hill parameters",
    "---------------",
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

outpath = Path(__file__).parent / "fragility_vs_xbar_logistic_multiple_cycles.png"
plt.savefig(outpath, dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved: {outpath}")