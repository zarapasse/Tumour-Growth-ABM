import numpy as np
from Models.Exponential_Model import TumourModel, HillParams
from Models.Exponential_Drug_Resistance import TumourModel as ResistantTumourModel


def process_config(config):
    """Extract and return simulation parameters from config dict."""
    sim = config["simulation"]
    hill = config.get("hill_parameters", None)

    initial_cells = sim["initial_cells"]
    birth_rate = sim["birth_rate"]
    death_rate = sim["death_rate"]
    dt = sim["dt"]
    steps = sim["steps"]
    n_runs = sim["n_runs"]
    p_mutation = sim["p_mutation"] if not None else 0.0
    initial_resistant_fraction = sim["initial_resistant_fraction"] if not None else 0.0
    fitness_cost = sim["fitness_cost"]

    hill_params = None
    if hill is not None:
        hill_params = HillParams(
            K_kill=hill["K_kill"],
            C=hill["C"],
            n=hill["n"],
        )

    return (
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
    )


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
          {"name": "Uneven Schedule",  "schedule": [...], "alpha": alpha},
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
        {"name": "Uneven Schedule", "schedule": odd_schedule, "alpha": alpha},
    ]


def run_abm(config, scenarios, seed=None, compute_fragility=False):
    """Run ABM for given scenarios. Returns mean/std trajectories and fragility if computed."""

    initial_cells, birth_rate, death_rate, dt, steps, n_runs, hill_params, _, _, _ = (
        process_config(config)
    )

    time = np.arange(steps + 1) * dt

    mean_trajectories = []
    std_trajectories = []
    final_volumes = []

    # reproducibility
    base_seed = 0 if seed is None else int(seed)
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
                alpha=sc["alpha"],
                drug_schedule=list(sc["schedule"]),
                hill_params=hill_params,
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

    results = {
        "scenarios": scenarios,
        "mean_trajectories": mean_trajectories,
        "std_trajectories": std_trajectories,
        "time": time,
    }

    if compute_fragility and len(final_volumes) >= 2:
        frag_per_run = (final_volumes[1] - final_volumes[0]) / initial_cells
        results["fragility"] = {
            "per_run": frag_per_run,
            "mean": frag_per_run.mean(),
            "std": frag_per_run.std(),
        }

    return results


def run_abm_resistance(
    config,
    scenarios,
    seed=None,
    compute_fragility=False,
    enable_resistance=True,
):
    """Run resistant ABM for given scenarios. Returns mean/std trajectories for Total, Sensitive, Resistant."""
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

    time = np.arange(steps + 1) * dt

    base_seed = 0 if seed is None else int(seed)
    run_seeds = [base_seed + r for r in range(n_runs)]

    mean_total, std_total = [], []
    mean_sens, std_sens = [], []
    mean_res, std_res = [], []
    final_totals = []

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
                alpha=float(sc["alpha"]),
                fitness_cost=fitness_cost,
                drug_schedule=list(sc["schedule"]),
                hill_params=hill_params,
                seed=run_seeds[r],
                enable_resistance=enable_resistance,
                p_mutation=p_mutation,
                initial_resistant_fraction=initial_resistant_fraction,
            )

            # Run steps
            for _ in range(steps):
                m.step()

            df = m.datacollector.get_model_vars_dataframe()
            # Ensure it matches steps+1
            all_total[r, :] = df["Total"].to_numpy()
            all_sens[r, :] = df["Sensitive"].to_numpy()
            all_res[r, :] = df["Resistant"].to_numpy()

        mean_total.append(all_total.mean(axis=0))
        std_total.append(all_total.std(axis=0))
        mean_sens.append(all_sens.mean(axis=0))
        std_sens.append(all_sens.std(axis=0))
        mean_res.append(all_res.mean(axis=0))
        std_res.append(all_res.std(axis=0))

        final_totals.append(all_total[:, -1])

    results = {
        "scenarios": scenarios,
        "time": time,
        "mean_total": mean_total,
        "std_total": std_total,
        "mean_sensitive": mean_sens,
        "std_sensitive": std_sens,
        "mean_resistant": mean_res,
        "std_resistant": std_res,
    }

    if compute_fragility and len(final_totals) >= 2:
        frag_per_run = (final_totals[1] - final_totals[0]) / float(initial_cells)
        results["fragility"] = {
            "per_run": frag_per_run,
            "mean": float(frag_per_run.mean()),
            "std": float(frag_per_run.std()),
        }

    return results


# ---------------------- Analytic solutions ---------------------- #
def hill_kill_rate(conc, hill_params):
    """
    Saturating Hill kill term:
        k_kill(c) = K_kill * x^n / (x^n + C^n)
    """
    K_kill = float(hill_params.K_kill)
    C = float(hill_params.C)
    n = float(hill_params.n)

    x_n = conc**n
    denominator = x_n + (C**n)

    return K_kill * (x_n / denominator)


def pk_concentration_series(time, schedule, alpha):
    """Exact PK concentration time series for given dose schedule and decay rate alpha."""

    time = np.asarray(time, dtype=float)
    conc = np.zeros_like(time, dtype=float)

    for amount, t_dose in schedule:
        amount = float(amount)
        t_dose = float(t_dose)
        mask = time >= t_dose
        conc[mask] += amount * np.exp(-alpha * (time[mask] - t_dose))

    return conc


def analytic_population_with_pk(
    time, N0, birth_rate, death_rate, schedule, alpha, hill_params
):
    """Deterministic expected population under time-varying kill from PK+Hill."""

    time = np.asarray(time, dtype=float)
    N = np.zeros_like(time, dtype=float)
    N[0] = float(N0)

    conc = pk_concentration_series(time, schedule, alpha)
    kill = hill_kill_rate(conc, hill_params)

    g = float(birth_rate) - (float(death_rate) + kill)

    for i in range(1, len(time)):
        dt_step = time[i] - time[i - 1]
        N[i] = N[i - 1] * np.exp(g[i - 1] * dt_step)

    return N, conc


def analytic_population_no_drug(time, N0, birth_rate, death_rate):
    """Closed-form exponential growth/decay with no drug."""
    return N0 * np.exp((birth_rate - death_rate) * time)
