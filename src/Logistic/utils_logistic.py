import numpy as np
from Logistic_Model import HillParams, TumourModel, ResourceParams

# from Logistic_Drug_Resistance import TumourResistantModel
from scipy.optimize import curve_fit
from scipy.integrate import solve_ivp


# -------------------- ABM + scenario utilities ------------------- #
def build_resource_params(res_dict):
    Emax = float(res_dict["energy_capacity"])
    influx = float(res_dict["resource_influx"])

    return ResourceParams(
        energy_capacity=Emax,
        resource_influx=influx,
        division_threshold=res_dict["frac_division_threshold"] * Emax,
        maintenance_cost=res_dict["frac_maintenance_cost"] * Emax,
    )


def process_config(config):
    sim_params = config["simulation"]
    hill = config.get("hill_parameters", None)
    res = config["resource_parameters"]
    initial_cells = sim_params["initial_cells"]
    birth_rate = sim_params["birth_rate"]
    death_rate = sim_params["death_rate"]
    dt = sim_params["dt"]
    steps = sim_params["steps"]
    n_runs = sim_params["n_runs"]
    initial_resources = sim_params["initial_resources"]
    p_mutation = sim_params["p_mutation"]
    initial_resistant_fraction = sim_params["initial_resistant_fraction"]
    initial_cell_energy = sim_params["initial_cell_energy"]

    hill_params = None
    if hill is not None:
        hill_params = HillParams(
            K_kill=hill["K_kill"],
            C=hill["C"],
            n=hill["n"],
        )

    res_params = build_resource_params(config["resource_parameters"])

    return (
        initial_cells,
        birth_rate,
        death_rate,
        dt,
        steps,
        n_runs,
        hill_params,
        res_params,
        initial_resources,
        initial_cell_energy,
        p_mutation,
        initial_resistant_fraction,
    )


# ------------------- Run ABM -------------------- #


def run_abm_logistic(config, scenarios, seed=None, compute_fragility=False):

    (
        initial_cells,
        birth_rate,
        death_rate,
        dt,
        steps,
        n_runs,
        hill_params,
        res_params,
        initial_resources,
        initial_cell_energy,
        _,
        _,
    ) = process_config(config)

    time = np.arange(steps + 1) * dt

    mean_trajectories = []
    std_trajectories = []
    final_volumes = []
    all_trajectories = []

    # reproducibility
    base_seed = seed if seed is not None else 0
    run_seeds = [base_seed + r for r in range(n_runs)]

    for sc in scenarios:
        all_counts = np.zeros((n_runs, steps + 1), dtype=float)

        for r in range(n_runs):
            print(f"Running scenario '{sc['name']}', run {r+1}/{n_runs}")

            m = TumourModel(
                initial_cells=initial_cells,
                birth_rate=birth_rate,
                death_rate=death_rate,
                dt=dt,
                initial_resources=initial_resources,
                initial_cell_energy=initial_cell_energy,
                res_params=res_params,
                alpha=sc["alpha"],
                hill_params=hill_params,
                drug_schedule=list(sc["schedule"]),
                seed=run_seeds[r],
            )

            counts = [len(m.agents)]
            for _ in range(steps):
                m.step()
                counts.append(len(m.agents))

            all_counts[r] = counts

        mean_trajectories.append(all_counts.mean(axis=0))
        std_trajectories.append(all_counts.std(axis=0))
        final_volumes.append(all_counts[:, -1])
        all_trajectories.append(all_counts)

    results = {
        "scenarios": scenarios,
        "mean_trajectories": mean_trajectories,
        "std_trajectories": std_trajectories,
        "all_trajectories": all_trajectories,
        "time": time,
    }

    if compute_fragility and len(final_volumes) >= 2:
        # same normalisation style as your exponential version
        frag_per_run = (final_volumes[1] - final_volumes[0]) / initial_cells
        results["fragility"] = {
            "per_run": frag_per_run,
            "mean": float(frag_per_run.mean()),
            "std": float(frag_per_run.std()),
        }

    return results


def make_fragility_test_scenarios(
    total_dose_per_cycle, n_doses_per_cycle, n_cycles, sigma, cycle_length, alpha
):
    """
    Build two scenarios (even vs uneven) over repeated treatment cycles.

    - Even: every dose in a cycle is mean_dose = total_dose_per_cycle / n_doses_per_cycle
    - Uneven: doses alternate mean_dose ± sigma, summing to the same total per cycle.

    Returns:
        [
          {"name": "Even Schedule", "schedule": [...], "alpha": alpha},
          {"name": "Odd Schedule",  "schedule": [...], "alpha": alpha},
        ]
    """

    mean_dose = total_dose_per_cycle / n_doses_per_cycle

    # dose times within a cycle
    dt_dose = cycle_length / n_doses_per_cycle

    # ---- even schedule ----
    even_schedule = []
    for c in range(n_cycles):
        cycle_start = c * cycle_length
        for i in range(n_doses_per_cycle):
            even_schedule.append((mean_dose, cycle_start + i * dt_dose))

    # ---- uneven schedule ----
    deviations = [sigma if i % 2 == 0 else -sigma for i in range(n_doses_per_cycle)]
    if n_doses_per_cycle % 2 == 1:
        deviations[-1] = 0.0  # keep per-cycle total exactly the same

    odd_schedule = []
    for c in range(n_cycles):
        cycle_start = c * cycle_length
        for i, dev in enumerate(deviations):
            odd_schedule.append((mean_dose + dev, cycle_start + i * dt_dose))

    return [
        {"name": "Even Schedule", "schedule": even_schedule, "alpha": alpha},
        {"name": "Odd Schedule", "schedule": odd_schedule, "alpha": alpha},
    ]


# ------------------- Logistic fit helpers ------------------- #
def logistic_function(t, K, r, N0):
    """Standard logistic growth curve."""
    N0 = max(N0, 1e-8)  # guard
    return K / (1 + ((K - N0) / N0) * np.exp(-r * t))


def fit_logistic_direct(t, N, K_guess=None):
    """Fit logistic function to (t, N) using scipy curve_fit."""
    if K_guess is None:
        K_guess = max(np.max(N) * 1.1, 1.0)
    N0_guess = max(N[0], 1e-6)
    r_guess = 0.1

    params, _ = curve_fit(
        logistic_function,
        t,
        N,
        p0=[K_guess, r_guess, N0_guess],
        bounds=([0, 0, 0], [np.inf, np.inf, np.inf]),
        maxfev=20000,
    )
    K, r, N0_fit = params
    fit = logistic_function(t, K, r, N0_fit)
    return K, r, N0_fit, fit


def compute_auc(trajs, dt):
    trajs = np.asarray(trajs, dtype=float)
    if trajs.ndim == 1:
        return float(np.trapz(trajs, dx=dt))
    return np.trapz(trajs, dx=dt, axis=1)


# ------------------- Continuum logistic + PK/PD comparator ------------------- #


def deterministic_pk(time, schedule, alpha):
    """
    Returns array conc(t) for given time array using:
    conc(t) = sum(amount * exp(-alpha * (t - t_dose))) for t >= t_dose
    """
    conc = np.zeros_like(time)

    for amount, t_dose in schedule:
        mask = time >= t_dose
        conc[mask] += amount * np.exp(-alpha * (time[mask] - t_dose))

    return conc


def hill_kill_rate_scalar(c, hill_params):
    """Scalar Hill kill term k(c)."""
    if hill_params is None or c <= 0.0:
        return 0.0
    K_kill = float(hill_params.K_kill)
    C = float(hill_params.C)
    n = float(hill_params.n)

    x_n = c**n
    denom = x_n + (C**n)
    if denom <= 0.0:
        return 0.0
    return float(K_kill * (x_n / denom))


# def continuum_logistic_with_pkpd(time, N0, K, r, schedule, alpha, hill_params,
#                                      method="RK45", rtol=1e-7, atol=1e-9):
#     """
#     Continuum logistic + drug solved with solve_ivp:

#         dN/dt = r N (1 - N/K) - k(c(t)) N

#     PK: conc(t) precomputed on `time` grid, then linearly interpolated during integration.
#     PD: k(c) from Hill equation.

#     Returns: N_sol, conc_grid, conc_used_grid, k_grid
#     """
#     time = np.asarray(time, dtype=float)
#     t0, t1 = float(time[0]), float(time[-1])

#     # PK on the same grid you plot
#     conc_grid = deterministic_pk(time, schedule, alpha)

#     # Linear interpolation of conc(t) using numpy.interp (fast, no extra deps)
#     def conc_of_t(t):
#         # clamp to [t0, t1] to avoid extrapolation weirdness
#         if t <= t0:
#             return float(conc_grid[0])
#         if t >= t1:
#             return float(conc_grid[-1])
#         return float(np.interp(t, time, conc_grid))

#     # ODE RHS
#     def rhs(t, y):
#         N = float(y[0])
#         # Optional: prevent negative N feeding back into dynamics
#         if N <= 0.0:
#             return [0.0]

#         c = conc_of_t(t)
#         k = hill_kill_rate_scalar(c, hill_params)
#         dNdt = r * N * (1.0 - N / K) - k * N
#         return [dNdt]

#     sol = solve_ivp(
#         rhs,
#         t_span=(t0, t1),
#         y0=[float(N0)],
#         t_eval=time,
#         method=method,
#         rtol=rtol,
#         atol=atol,
#         vectorized=False,
#     )

#     if not sol.success:
#         raise RuntimeError(f"solve_ivp failed: {sol.message}")

#     N_sol = sol.y[0]
#     # keep outputs nonnegative for plotting/comparison
#     N_sol = np.maximum(N_sol, 0.0)

#     # For plotting the same “used” profiles on the grid
#     conc_used = conc_grid
#     k_grid = hill_kill_rate(conc_used, hill_params)

#     return N_sol, conc_grid, conc_used, k_grid

# def continuum_logistic_with_pkpd(time, dt, N0, K, r, schedule, alpha, hill_params):
#     """
#     Continuum logistic + drug:
#         dN/dt = r N (1 - N/K) - k(x(t)) N

#     PK: conc(t) from deterministic_pk(time, schedule, alpha)
#     PD: k(conc) from hill_kill_rate, matching the ABM Hill parameters.

#     Uses conc(t) at the current time grid (no one-step lag), matching the edited ABM.
#     """
#     conc = deterministic_pk(time, schedule, alpha)
#     conc_used = conc

#     k = hill_kill_rate(conc_used, hill_params)

#     N = np.zeros_like(time, dtype=float)
#     N[0] = float(N0)

#     for j in range(1, len(time)):
#         Nj = N[j - 1]
#         growth = r * Nj * (1.0 - Nj / K)
#         kill = k[j - 1] * Nj
#         N[j] = max(0.0, Nj + dt * (growth - kill))

#     return N, conc, conc_used, k


def hill_kill_rate(conc, hill_params):
    """
    Saturating Hill kill term:
        k_kill(c) = K_kill * c^n / (c^n + C^n)
    """
    K_kill = float(hill_params.K_kill)
    C = float(hill_params.C)
    n = float(hill_params.n)

    x_n = np.asarray(conc, dtype=float) ** n
    denom = x_n + (C**n)

    return K_kill * (x_n / np.maximum(denom, 1e-12))


# ---------------- Utility helpers (Option B) ---------------- #
def estimate_K_tail(N, frac_tail=0.2):
    N = np.asarray(N, float)
    tail = N[int((1 - frac_tail) * len(N)) :]
    return float(np.mean(tail))


def fit_r_logit(time, N, K, low_frac=0.2, high_frac=0.8, eps=1e-6):
    """
    Fit r from log(N/(K-N)) = r t + c, using points where N is in [low_frac*K, high_frac*K]
    to avoid near-0 and near-K noise.
    """
    t = np.asarray(time, float)
    y = np.asarray(N, float)

    lo = low_frac * K
    hi = high_frac * K
    m = (y > lo) & (y < hi)

    if m.sum() < 6:
        raise ValueError(
            "Not enough points in the mid-range to identify r. Adjust low/high_frac."
        )

    y_clip = np.clip(y[m], eps, K - eps)
    z = np.log(y_clip / (K - y_clip))
    r, c = np.polyfit(t[m], z, 1)
    return float(r), float(c)


import numpy as np
from scipy.integrate import solve_ivp


def pk_conc_analytic(t, schedule, alpha):
    """
    Exact PK concentration at time t from bolus doses with exponential decay.
    schedule entries are (amount, t_dose).
    """
    if not schedule:
        return 0.0

    if alpha == 0.0:
        # no decay => stepwise accumulation
        return float(sum(amount for amount, t_dose in schedule if t >= t_dose))

    total = 0.0
    for amount, t_dose in schedule:
        if t >= t_dose:
            total += amount * np.exp(-alpha * (t - t_dose))
    return float(total)


def hill_kill_rate_scalar(c, hill_params):
    """Scalar Hill kill term k(c)."""
    if hill_params is None or c <= 0.0:
        return 0.0
    K_kill = float(hill_params.K_kill)
    C = float(hill_params.C)
    n = float(hill_params.n)

    x_n = c**n
    denom = x_n + (C**n)
    if denom <= 0.0:
        return 0.0
    return float(K_kill * (x_n / denom))


def continuum_logistic_with_pkpd(
    time,
    N0,
    K,
    r,
    schedule,
    alpha,
    hill_params,
    method="RK45",
    rtol=1e-7,
    atol=1e-9,
):
    """
    Logistic + PK/PD solved with solve_ivp, computing PK analytically inside RHS.

        dN/dt = r N (1 - N/K) - k(c(t)) N

    Returns:
        N_sol (len(time)),
        conc_grid (len(time))  -- conc(t) evaluated on time grid for plotting
        conc_used_grid (same as conc_grid here),
        k_grid (len(time))     -- kill term evaluated on grid for plotting
    """
    time = np.asarray(time, dtype=float)
    t0, t1 = float(time[0]), float(time[-1])

    # RHS for solve_ivp
    def rhs(t, y):
        N = float(y[0])
        if N <= 0.0:
            return [0.0]  # keep it at 0 once extinct

        c = pk_conc_analytic(t, schedule, alpha)
        k = hill_kill_rate_scalar(c, hill_params)

        dNdt = r * N * (1.0 - N / K) - k * N
        return [dNdt]

    sol = solve_ivp(
        rhs,
        t_span=(t0, t1),
        y0=[float(N0)],
        t_eval=time,
        method=method,
        rtol=rtol,
        atol=atol,
    )

    if not sol.success:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")

    N_sol = np.maximum(sol.y[0], 0.0)

    # For plotting panel (B) and optional debugging
    conc_grid = np.array(
        [pk_conc_analytic(t, schedule, alpha) for t in time], dtype=float
    )
    k_grid = np.array(
        [hill_kill_rate_scalar(c, hill_params) for c in conc_grid], dtype=float
    )

    return N_sol, conc_grid, conc_grid, k_grid
