"""
TGI-based fragility sweep (resource-dependent / logistic ABM) + deterministic logistic comparator

We use time-integrated tumour burden (TGI) instead of endpoint size:
    TGI = ∫_0^T N(t) dt  (approximated numerically on the simulation grid)

Define "transient fragility" using TGI:
    F_TGI = (TGI_odd - TGI_even) / TGI0
where TGI0 is the no-treatment baseline over the same horizon.

This script computes:
- ABM: mean ± SD of F_TGI across stochastic runs
- Deterministic: F_TGI_det from the fitted logistic+PK/PD ODE

Key detail: logistic parameters (K,r) are fit using a WINDOWED fit (time >= t_start).
"""

import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from utils_logistic import (
    run_abm_logistic,
    make_fragility_test_scenarios,
    process_config,
    deterministic_pk,               # needed for continuum comparator
)

# =========================
# Logistic fit (WINDOWED)
# =========================
def logistic_function(t, K, r, N0):
    return K / (1 + ((K - N0) / N0) * np.exp(-r * t))

def logistic_fit_windowed(time, mean_cells, t_start=10.0):
    """
    Fit logistic curve using only time >= t_start, but return the fitted curve
    over the FULL time array for plotting.

    Notes:
    - We shift time so exponentials are numerically well-conditioned.
    - Returned N0 is the logistic initial condition at the fit window start (t_start),
      not necessarily mean_cells[0].
    """
    time = np.asarray(time, dtype=float)
    mean_cells = np.asarray(mean_cells, dtype=float)

    mask = time >= t_start
    if mask.sum() < 5:
        raise RuntimeError(f"Not enough points to fit after t_start={t_start}.")

    t_fit = time[mask]
    y_fit = mean_cells[mask]

    # shift time so exponentials are well-conditioned
    t0 = t_fit[0]
    t_fit_shift = t_fit - t0

    # initial guesses
    K_guess = np.max(y_fit) * 1.05
    r_guess = 0.1
    N0_guess = max(float(y_fit[0]), 1e-6)

    params, _ = curve_fit(
        logistic_function,
        t_fit_shift,
        y_fit,
        p0=[K_guess, r_guess, N0_guess],
        bounds=([0, 0, 0], [np.inf, np.inf, np.inf]),
        maxfev=20000,
    )
    K, r, N0 = params

    # build fitted curve over full time array (using same time shift convention)
    full_shift = time - t0
    fitted_full = logistic_function(full_shift, K, r, N0)

    return float(K), float(r), float(N0), fitted_full

# =========================
# Hill + continuum logistic
# =========================
def hill_equation(drug_conc, hill_params):
    """
    Match TumourModel.hill_equation exactly:
    H(x) = E0 + (x^n (E1 - E0)) / (x^n + C^n)
    """
    E0 = hill_params["E0"]
    E1 = hill_params["E1"]
    C  = hill_params["C"]
    n  = hill_params["n"]

    drug_conc = np.asarray(drug_conc, dtype=float)

    kill = np.empty_like(drug_conc)
    kill[:] = E0

    pos = drug_conc > 0
    x = drug_conc[pos]
    kill[pos] = E0 + (x**n * (E1 - E0)) / (x**n + C**n)

    return kill

def continuum_logistic_with_pkpd(time, dt, N0, K, r, schedule, alpha, hill_params, lag_one_step=True):
    """
    Continuum logistic + drug:
        dN/dt = r N (1 - N/K) - k(x(t)) N

    Uses deterministic_pk(time, schedule, alpha) for PK
    Uses hill_equation(...) matched to your ABM for PD

    lag_one_step=True replicates your ABM convention where p_death uses the previous
    timestep's stored drug_conc ("current_drug_conc").
    """
    time = np.asarray(time, dtype=float)

    # PK concentration profile
    conc = deterministic_pk(time, schedule, alpha)

    # ABM one-step lag convention
    if lag_one_step:
        conc_used = np.zeros_like(conc)
        conc_used[1:] = conc[:-1]
    else:
        conc_used = conc

    # PD kill rate
    k = hill_equation(conc_used, hill_params)

    N = np.zeros_like(time, dtype=float)
    N[0] = float(N0)

    for j in range(1, len(time)):
        Nj = N[j - 1]
        growth = r * Nj * (1.0 - Nj / K)
        kill   = k[j - 1] * Nj
        N[j] = max(0.0, Nj + dt * (growth - kill))

    return N, conc, conc_used, k

# =========================
# Helpers
# =========================
def set_horizon_steps(cfg, t_end):
    dt_local = cfg["simulation"]["dt"]
    cfg["simulation"]["steps"] = int(t_end / dt_local) + 1

def compute_tgi(trajs, dt):
    """
    trajs: array (n_runs, steps) OR (steps,)
    Returns:
      - (n_runs,) if input is 2D
      - scalar if input is 1D
    """
    trajs = np.asarray(trajs, dtype=float)
    if trajs.ndim == 1:
        return float(np.trapezoid(trajs, dx=dt))
    return np.trapezoid(trajs, dx=dt, axis=1)

# =========================
# Load config
# =========================
CONFIG_PATH = Path(__file__).parent / "config_logistic.json"
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

# =========================
# User sweep settings
# =========================
x_bar_values = np.arange(10, 100, 5)

n_doses_per_cycle = 2
n_cycles = 4
cycle_length = 12
alpha_val = 1.0

# initialise near carrying capacity (as a fraction of K from no-drug fit)
target_frac_K = 0.85

# window for logistic fit
fit_window_start = 10.0

base_seed = 42

# ============================================================
# 1) Estimate K,r from a LONG no-drug run (WINDOWED FIT)
# ============================================================
t_end_baseline = max(10 * cycle_length, config["simulation"]["steps"] * config["simulation"]["dt"])
set_horizon_steps(config, t_end_baseline)

initial_cells, birth_rate, death_rate, dt, steps, n_runs, hill_params = process_config(config)
time_baseline = np.arange(steps) * dt

baseline_scenario = [{"name": "No Drug", "schedule": [], "alpha": 0.0}]
baseline_out = run_abm_logistic(config, baseline_scenario, seed=base_seed)

mean_no_drug = baseline_out["mean_trajectories"][0]

K0, r0, N0_fit_at_window, baseline_fit = logistic_fit_windowed(
    time_baseline, mean_no_drug, t_start=fit_window_start
)

print(f"[No-drug windowed fit] t_start={fit_window_start} | K={K0:.3f}, r={r0:.4f}, N0@window={N0_fit_at_window:.2f}")

# Set initial condition near K (based on fitted K)
config["simulation"]["initial_cells"] = int(target_frac_K * K0)

# ============================================================
# 2) Set horizon for the ACTUAL sweep
# ============================================================
t_end = n_cycles * cycle_length
set_horizon_steps(config, t_end)

initial_cells, birth_rate, death_rate, dt, steps, n_runs, hill_params = process_config(config)
time = np.arange(steps) * dt

# ============================================================
# 3) Compute TGI0 baseline for normalization (same horizon!)
# ============================================================
baseline_out2 = run_abm_logistic(config, baseline_scenario, seed=base_seed)
baseline_trajs = baseline_out2["all_trajectories"][0]  # (n_runs, steps)
TGI0_per_run = compute_tgi(baseline_trajs, dt)
TGI0_mean = float(TGI0_per_run.mean())

# deterministic no-drug baseline (for deterministic TGI0)
N0_det = float(initial_cells)
N_det0, conc0, conc0_used, k0 = continuum_logistic_with_pkpd(
    time=time,
    dt=dt,
    N0=N0_det,
    K=K0,
    r=r0,
    schedule=[],
    alpha=0.0,
    hill_params=hill_params,
    lag_one_step=True,
)
TGI0_det = compute_tgi(N_det0, dt)

# ============================================================
# 4) Sweep x_bar and compute F_TGI (ABM + deterministic)
# ============================================================
F_TGI_mean = []
F_TGI_std  = []
F_TGI_det  = []

for j, x_bar in enumerate(x_bar_values):
    print(f"Computing TGI fragility for x_bar = {x_bar}...")

    total_dose_per_cycle = x_bar * n_doses_per_cycle

    # maximal uneven for n_doses=2 => sigma = mean dose = x_bar (holiday dosing)
    sigma = x_bar

    scenarios = make_fragility_test_scenarios(
        total_dose=total_dose_per_cycle,
        n_doses=n_doses_per_cycle,
        n_cycles=n_cycles,
        sigma=sigma,
        cycle_length=cycle_length,
        alpha=alpha_val,
    )

    # ---------- ABM ----------
    out = run_abm_logistic(config, scenarios, seed=base_seed + j)

    even_trajs = out["all_trajectories"][0]  # (n_runs, steps)
    odd_trajs  = out["all_trajectories"][1]

    TGI_even = compute_tgi(even_trajs, dt)   # (n_runs,)
    TGI_odd  = compute_tgi(odd_trajs, dt)

    F_TGI = (TGI_odd - TGI_even) / TGI0_mean

    F_TGI_mean.append(float(F_TGI.mean()))
    F_TGI_std.append(float(F_TGI.std(ddof=1)) if F_TGI.size > 1 else 0.0)

    # ---------- Deterministic ----------
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

    TGI_even_det = compute_tgi(N_even, dt)
    TGI_odd_det  = compute_tgi(N_odd, dt)

    F_TGI_det.append(float((TGI_odd_det - TGI_even_det) / TGI0_det))

F_TGI_mean = np.array(F_TGI_mean, dtype=float)
F_TGI_std  = np.array(F_TGI_std, dtype=float)
F_TGI_det  = np.array(F_TGI_det, dtype=float)

# ============================================================
# 5) Plot
# ============================================================
fig, ax = plt.subplots(figsize=(12, 7))

ax.plot(x_bar_values, F_TGI_mean, lw=2, label="ABM mean", color="blue")
ax.fill_between(
    x_bar_values,
    F_TGI_mean - F_TGI_std,
    F_TGI_mean + F_TGI_std,
    alpha=0.15,
    label="ABM ± SD",
    color="blue",
)

ax.plot(x_bar_values, F_TGI_det, lw=2, color="black", label="Logistic + PK/PD")

ax.axhline(0, ls="--", lw=1)

ax.set_title("TGI-based fragility vs mean dose (resource-dependent ABM - 4 cycles)")
ax.set_xlabel(r"Mean dose $\bar{x}$ (mg/L)")
ax.set_ylabel("TGI fragility")
ax.grid(True, alpha=0.3)
ax.legend()

# ---------------- Side parameter panel ---------------- #
panel_lines = [
    "Fragility sweep",
    "----------------",
    f"alpha         = {alpha_val}",
    f"n_cycles      = {n_cycles}",
    f"cycle_length  = {cycle_length}",
    f"doses/cycle   = {n_doses_per_cycle}",
    f"horizon T     = {t_end}",
    "",
    "No-drug logistic fit",
    "---------------------",
    f"K ≈ {K0:.2f}",
    f"r ≈ {r0:.4f}",
    "",
    "ABM parameters",
    "---------------",
    f"birth_rate     = {birth_rate}",
    f"death_rate     = {death_rate}",
    f"dt, steps      = {dt}, {steps}",
    f"n_runs         = {n_runs}",
    "",
    "Hill parameters",
    "----------------",
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

outpath = Path(__file__).parent / "tgi_fragility_logistic_4_cycle.png"
plt.savefig(outpath, dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved: {outpath}")