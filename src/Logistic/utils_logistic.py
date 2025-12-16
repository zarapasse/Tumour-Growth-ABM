import numpy as np
from Logistic_Drug_Model import TumourModel
from scipy.optimize import curve_fit


def run_abm_logistic(config, scenarios, seed=1, compute_fragility=False):
    """
    Run the energy-based logistic ABM under multiple dosing scenarios.

    Returns dict with:
        - mean_trajectories
        - std_trajectories
        - all_trajectories
        - fragility (if compute_fragility=True)
    """
    np.random.seed(seed)

    sim = config["simulation"]

    # Load model parameters
    initial_cells = sim["initial_cells"]
    birth_rate = sim["birth_rate"]
    death_rate = sim["death_rate"]
    dt = sim["dt"]
    steps = sim["steps"]
    n_runs = sim["n_runs"]

    initial_resources = sim["initial_resources"]
    resource_influx = sim["resource_influx"]
    energy_capacity = sim["energy_capacity"]

    hill_params = config["hill_parameters"]

    all_means = []
    all_stds = []
    all_runs_list = []

    # For fragility: store total cells per scenario across runs
    scenario_final_sizes = []

    for sc in scenarios:

        schedule = sc["schedule"]
        alpha = sc["alpha"]

        runs = []
        for r in range(n_runs):

            model = TumourModel(
                initial_cells=initial_cells,
                birth_rate=birth_rate,
                death_rate=death_rate,
                dt=dt,
                initial_resources=initial_resources,
                resource_influx=resource_influx,
                energy_capacity=energy_capacity,
                alpha=alpha,
                drug_schedule=schedule,
                hill_params=hill_params,
            )

            traj = []
            for _ in range(steps):
                model.step()
                traj.append(len(model.agents))

            runs.append(traj)

        runs = np.array(runs)
        mean = runs.mean(axis=0)
        std = runs.std(axis=0)

        all_means.append(mean)
        all_stds.append(std)
        all_runs_list.append(runs)

        scenario_final_sizes.append(runs[:, -1])  # last timestep

    # ----- fragility calculation -----
    if compute_fragility:
        if len(scenarios) != 2:
            raise ValueError(
                "Fragility test needs exactly TWO scenarios (even/uneven)."
            )

        # fragility = mean(N_odd - N_even)
        frag = scenario_final_sizes[1] - scenario_final_sizes[0]

        return {
            "mean_trajectories": all_means,
            "std_trajectories": all_stds,
            "all_trajectories": all_runs_list,
            "fragility": {
                "mean": np.mean(frag),
                "per_run": frag,
            },
        }

    return {
        "mean_trajectories": all_means,
        "std_trajectories": all_stds,
        "all_trajectories": all_runs_list,
    }


def logistic_function(t, K, r, N0):
    """Standard logistic ODE solution."""
    return K / (1 + ((K - N0) / N0) * np.exp(-r * t))


def logistic_fit(time, mean_cells):
    """
    Fit logistic curve to the ABM mean trajectory.
    Returns (K, r, N0, fitted_curve).
    """
    K_guess = np.max(mean_cells) * 1.2
    r_guess = 0.1
    N0_guess = mean_cells[0]

    try:
        params, _ = curve_fit(
            logistic_function,
            time,
            mean_cells,
            p0=[K_guess, r_guess, N0_guess],
            bounds=([0, 0, 0], [np.inf, np.inf, np.inf]),
            maxfev=10000,
        )
        K, r, N0 = params
        return K, r, N0, logistic_function(time, K, r, N0)
    except Exception:
        return None, None, None, None


def analytic_logistic_with_pk(time, mean_cells):
    """
    For the logistic energy-based ABM,
    the “analytic” comparator is simply the logistic fit.

    This mirrors analytic_population_with_pk() in exponential ABM.
    """
    K, r, N0, fitted = logistic_fit(time, mean_cells)
    return fitted


def make_fragility_test_scenarios(
    total_dose, n_doses, n_cycles, sigma, cycle_length, alpha
):
    """
    Create two ABM scenarios for fragility analysis: even and uneven schedules.

    Parameters:
        total_dose (float): total dose per cycle
        n_doses (int): number of doses per cycle
        n_cycles (int): number of repeated cycles
        sigma (float): additive deviation from mean for uneven schedule (>=0)
        cycle_length (float): length of one cycle (time units)
        alpha (float): PK decay rate (kept for API compatibility)

    Returns:
        list of dict: two scenario dicts (even and odd/uneven) ready for ABM
    """

    if n_doses <= 0:
        raise ValueError("n_doses must be >= 1")

    mean_dose = total_dose / n_doses
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    if sigma > mean_dose:
        raise ValueError(
            f"sigma is too large (would produce negative doses). "
            f"Require sigma <= mean_dose ({mean_dose})."
        )

    # ---- Even schedule ----
    even_schedule = []
    for c in range(n_cycles):
        cycle_start = c * cycle_length
        dose_amount = mean_dose
        times = [cycle_start + i * (cycle_length / n_doses) for i in range(n_doses)]
        even_schedule += [(dose_amount, t) for t in times]

    # ---- Uneven / odd schedule using additive sigma ----
    # Build deviations that sum to zero: +sigma, -sigma, +sigma, -sigma, ...
    # If n_doses is odd, set the last deviation to 0 to keep sum exactly zero.
    deviations = [sigma if i % 2 == 0 else -sigma for i in range(n_doses)]
    if n_doses % 2 == 1:
        deviations[-1] = 0.0

    # Scaled doses = mean + deviation (guaranteed non-negative by check above)
    odd_schedule = []
    for c in range(n_cycles):
        cycle_start = c * cycle_length
        times = [cycle_start + i * (cycle_length / n_doses) for i in range(n_doses)]
        doses = [mean_dose + d for d in deviations]
        odd_schedule += list(zip(doses, times))

    # ---- Package as ABM-ready scenario dicts ----
    scenarios = [
        {"name": "Even Schedule", "schedule": even_schedule, "alpha": alpha},
        {"name": "Odd Schedule", "schedule": odd_schedule, "alpha": alpha},
    ]

    return scenarios


def process_config(config):
    sim_params = config["simulation"]
    hill_params = config["hill_parameters"]
    initial_cells = sim_params["initial_cells"]
    birth_rate = sim_params["birth_rate"]
    death_rate = sim_params["death_rate"]
    dt = sim_params["dt"]
    steps = sim_params["steps"]
    n_runs = sim_params["n_runs"]

    return initial_cells, birth_rate, death_rate, dt, steps, n_runs, hill_params


def process_scenarios_config(config):
    drug_params = config["drug"]
    scenarios = []
    if "schedule" in drug_params and isinstance(drug_params["schedule"], list):
        for idx, sc in enumerate(drug_params["schedule"]):
            name = sc.get("name", f"Schedule {idx+1}")
            alpha = sc.get("alpha", drug_params.get("alpha"))
            sched = [tuple(d) for d in sc.get("schedule", [])]
            scenarios.append({"name": name, "schedule": sched, "alpha": alpha})
    return scenarios


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
