import numpy as np
from Models.Logistic_Model import HillParams, TumourModel, ResourceParams

from Models.Logistic_Drug_Resistance import ResistantTumourModel
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
    fitness_cost = sim_params["fitness_cost"]

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
        fitness_cost,
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
        frag_per_run = (final_volumes[1] - final_volumes[0]) / initial_cells
        results["fragility"] = {
            "per_run": frag_per_run,
            "mean": float(frag_per_run.mean()),
            "std": float(frag_per_run.std()),
        }

    return results


def run_abm_resistant_logistic(config, scenarios, seed=None, compute_fragility=False):

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
        p_mutation,
        initial_resistant_fraction,
        fitness_cost,
    ) = process_config(config)

    time = np.arange(steps + 1) * dt

    mean_total, std_total = [], []
    mean_sens, std_sens = [], []
    mean_res, std_res = [], []
    final_totals = []

    all_total_list = []
    all_sens_list = []
    all_res_list = []

    # reproducibility
    base_seed = seed if seed is not None else 0
    run_seeds = [base_seed + r for r in range(n_runs)]

    for sc in scenarios:
        all_total = np.zeros((n_runs, steps + 1), dtype=float)
        all_sens = np.zeros((n_runs, steps + 1), dtype=float)
        all_res = np.zeros((n_runs, steps + 1), dtype=float)

        for r in range(n_runs):
            print(f"Running scenario '{sc['name']}', run {r+1}/{n_runs}")

            m = ResistantTumourModel(
                initial_cells=initial_cells,
                birth_rate=birth_rate,
                death_rate=death_rate,
                dt=dt,
                initial_resources=initial_resources,
                initial_cell_energy=initial_cell_energy,
                p_mutation=p_mutation,
                initial_resistant_fraction=initial_resistant_fraction,
                fitness_cost=fitness_cost,
                res_params=res_params,
                alpha=sc["alpha"],
                hill_params=hill_params,
                drug_schedule=list(sc["schedule"]),
                seed=run_seeds[r],
            )

            for _ in range(steps):
                m.step()

            df = m.datacollector.get_model_vars_dataframe()

            tcol = df["t"].to_numpy(dtype=float)
            if len(tcol) != steps + 1:
                raise ValueError(
                    f"Expected {steps+1} rows, got {len(tcol)}. Check DataCollector timing."
                )
            if not np.allclose(tcol, time):
                raise ValueError(
                    "Time grid mismatch.\n"
                    f"df['t'] head={tcol[:5]}, tail={tcol[-5:]}\n"
                    f"expected head={time[:5]}, tail={time[-5:]}"
                )

            all_total[r, :] = df["Total"].to_numpy(dtype=float)
            all_sens[r, :] = df["Sensitive"].to_numpy(dtype=float)
            all_res[r, :] = df["Resistant"].to_numpy(dtype=float)

        # summary stats
        mean_total.append(all_total.mean(axis=0))
        std_total.append(all_total.std(axis=0))
        mean_sens.append(all_sens.mean(axis=0))
        std_sens.append(all_sens.std(axis=0))
        mean_res.append(all_res.mean(axis=0))
        std_res.append(all_res.std(axis=0))

        final_totals.append(all_total[:, -1])

        all_total_list.append(all_total)
        all_sens_list.append(all_sens)
        all_res_list.append(all_res)

    results = {
        "scenarios": scenarios,
        "time": time,
        "mean_total": mean_total,
        "std_total": std_total,
        "mean_sensitive": mean_sens,
        "std_sensitive": std_sens,
        "mean_resistant": mean_res,
        "std_resistant": std_res,
        "all_total": all_total_list,
        "all_sensitive": all_sens_list,
        "all_resistant": all_res_list,
    }

    if compute_fragility and len(final_totals) >= 2:
        frag_per_run = (final_totals[1] - final_totals[0]) / float(initial_cells)
        results["fragility"] = {
            "per_run": frag_per_run,
            "mean": float(frag_per_run.mean()),
            "std": float(frag_per_run.std(ddof=1)) if frag_per_run.size > 1 else 0.0,
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
        k = hill_kill_rate(c, hill_params)

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
    k_grid = np.array([hill_kill_rate(c, hill_params) for c in conc_grid], dtype=float)

    return N_sol, conc_grid, conc_grid, k_grid


def pk_conc(t, schedule, alpha):
    t = np.asarray(t)
    conc = np.zeros_like(t, dtype=float)

    for amount, t_dose in schedule:
        dt = t - t_dose
        conc += amount * np.exp(-alpha * dt) * (dt >= 0)

    return conc


def plot_composition(ax, res, title, time):
    sens = res["mean_sensitive"]
    resi = res["mean_resistant"]
    total = sens + resi
    frac = resi / np.maximum(1.0, total)

    # --- Stackplot ---
    ax.stackplot(
        time,
        sens,
        resi,
        labels=["Sensitive", "Resistant"],
        colors=["tab:blue", "tab:red"],
        alpha=0.70,
        zorder=1,
    )

    ax.set_title(title)
    ax.set_ylabel("Cells")
    ax.grid(alpha=0.3)

    # --- Twin axis for resistant fraction ---
    ax2 = ax.twinx()
    ax2.plot(
        time,
        frac,
        color="0.35",      # grey
        ls="--",           # dashed
        lw=2.2,
        zorder=10,
        label="Resistant fraction",
    )
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("")
    ax2.tick_params(axis="y", labelsize=9)

    # --- Combined legend BELOW the axis ---
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()

    ax.legend(
        h1 + h2,
        l1 + l2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),   # below axis
        ncol=3,
        fontsize=8,
        frameon=False,
        handlelength=2.2,
        columnspacing=1.2,
    )