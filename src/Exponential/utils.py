import random
import numpy as np
from Exponential_Model import TumourModel
from Exponential_Drug_Resistance import TumourModel as ResistantTumourModel


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
            scenarios.append({
                "name": name,
                "schedule": sched,
                "alpha": alpha
            })
    return scenarios

# Create schedules for testing multiple cycles

def make_fragility_test_scenarios(total_dose, n_doses, n_cycles, sigma, cycle_length, alpha):
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
        raise ValueError(f"sigma is too large (would produce negative doses). "
                         f"Require sigma <= mean_dose ({mean_dose}).")

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
        {"name": "Odd Schedule", "schedule": odd_schedule, "alpha": alpha}
    ]

    return scenarios

def calculate_fragility(results_even, results_odd, initial_cells):
    """
    Calculate fragility between an even and odd schedule.

    Parameters:
        results_even (np.array): shape (n_runs, steps) or (steps,) if mean, ABM results for even schedule
        results_odd (np.array): same as above, for odd schedule
        initial_cells (int): initial tumour cell number

    Returns:
        float: mean fragility
        np.array: fragility per run (if multiple runs)
    """
    # Convert to shape (n_runs, steps) if only mean passed
    if results_even.ndim == 1:
        results_even = results_even.reshape(1, -1)
    if results_odd.ndim == 1:
        results_odd = results_odd.reshape(1, -1)


    fragility_per_run = (results_odd[:, -1] - results_even[:, -1]) / initial_cells

    mean_fragility = fragility_per_run.mean()

    return mean_fragility, fragility_per_run







#---------------------- ABM simulation runner ---------------------- #
def run_abm_for_schedule(config, schedule, alpha_val, seed=None):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    initial_cells, birth_rate, death_rate, dt, steps, n_runs, hill_params, = process_config(config)
    all_counts = np.zeros((n_runs, steps), dtype=float)
    conc_trace = None
    for r in range(n_runs):
        m = TumourModel(
            initial_cells,
            birth_rate,
            death_rate,
            dt,
            drug_schedule=schedule.copy(),
            alpha=alpha_val,
            hill_params=hill_params,
        )
        counts = []
        concs = []
        for _ in range(steps):
            m.step()
            counts.append(len(m.agents))
            concs.append(m.drug_conc)
        all_counts[r] = counts
        if r == 0:
            conc_trace = np.array(concs, dtype=float)
    mean = all_counts.mean(axis=0)
    std = all_counts.std(axis=0)
    return mean, std, conc_trace




def run_abm(config, scenarios, seed=None, compute_fragility=False):
    """
    Run ABM for given scenarios. Optionally compute fragility if two scenarios (even/odd) are provided.

    Parameters:
        config (dict): ABM config
        scenarios (list of dict): each dict with keys 'name', 'schedule', 'alpha'
        seed (int, optional): random seed
        compute_fragility (bool): if True, calculate fragility between first two scenarios
    Returns:
        dict: results including mean/std trajectories, optionally fragility
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    initial_cells, birth_rate, death_rate, dt, steps, n_runs, hill_params = process_config(config)

    mean_trajectories = []
    std_trajectories = []
    final_volumes = []

    for sc in scenarios:
        all_counts = np.zeros((n_runs, steps), dtype=float)
        for r in range(n_runs):
            print(f"Running scenario '{sc['name']}', run {r+1}/{n_runs}")
            m = TumourModel(
                initial_cells,
                birth_rate,
                death_rate,
                dt,
                drug_schedule=sc["schedule"].copy(),
                alpha=sc["alpha"],
                hill_params=hill_params,
            )
            counts = []
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
        "std_trajectories": std_trajectories
    }

    # Compute fragility if requested and we have two schedules
    if compute_fragility and len(final_volumes) >= 2:
        frag_per_run = (final_volumes[1] - final_volumes[0]) / initial_cells
        results["fragility"] = {
            "per_run": frag_per_run,
            "mean": frag_per_run.mean(),
            "std": frag_per_run.std()
        }

    return results


def run_abm_for_resistant(config, scenarios, seed=None, compute_fragility=False):
    """
    Run ABM for given scenarios. Optionally compute fragility if two scenarios (even/odd) are provided.

    Parameters:
        config (dict): ABM config
        scenarios (list of dict): each dict with keys 'name', 'schedule', 'alpha'
        seed (int, optional): random seed
        compute_fragility (bool): if True, calculate fragility between first two scenarios
    Returns:
        dict: results including mean/std trajectories, optionally fragility
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    initial_cells, birth_rate, death_rate, dt, steps, n_runs, hill_params = process_config(config)

    mean_trajectories = []
    std_trajectories = []
    final_volumes = []

    for sc in scenarios:
        all_counts = np.zeros((n_runs, steps, 2), dtype=float)
        for r in range(n_runs):
            print(f"Running scenario '{sc['name']}', run {r+1}/{n_runs}")
            m = ResistantTumourModel(
                initial_cells,
                birth_rate,
                death_rate,
                dt,
                drug_schedule=sc["schedule"].copy(),
                alpha=sc["alpha"],
                hill_params=hill_params,
                p_mutation=config["simulation"]["p_mutation"]
            )

            for _ in range(steps):
                m.step()
                
            df = m.datacollector.get_model_vars_dataframe()
            all_counts[r, :, 0] = df["Sensitive"].values[:steps]
            all_counts[r, :, 1] = df["Resistant"].values[:steps]



        mean_trajectories.append(all_counts.mean(axis=0))
        std_trajectories.append(all_counts.std(axis=0))
        final_volumes.append(all_counts[:, -1])

    results = {
        "scenarios": scenarios,
        "mean_trajectories": mean_trajectories,
        "std_trajectories": std_trajectories
    }

    # Compute fragility if requested and we have two schedules
    if compute_fragility and len(final_volumes) >= 2:
        frag_per_run = (final_volumes[1] - final_volumes[0]) / initial_cells
        results["fragility"] = {
            "per_run": frag_per_run,
            "mean": frag_per_run.mean(),
            "std": frag_per_run.std()
        }

    return results











# ---------------------- Analytic solutions ---------------------- #
def hill_effect(x, n, C, E0, E1):
    return E0 + (x**n * (E1 - E0)) / (x**n + C**n + 1e-12)

def pk_concentration_series(t, schedule, alpha):
    conc = np.zeros_like(t, dtype=float)
    for amount, t_dose in schedule:
        mask = t >= t_dose
        conc[mask] += amount * np.exp(-alpha * (t[mask] - t_dose))
    return conc

def analytic_population_with_pk(time, N0, birth_rate, death_rate, schedule, alpha, hp):
    conc = pk_concentration_series(time, schedule, alpha)
    H = hill_effect(conc, hp["n"], hp["C"], hp["E0"], hp["E1"])
    g = birth_rate - (death_rate + H)
    N = np.zeros_like(time, dtype=float)
    N[0] = N0
    for i in range(1, len(time)):
        dt_step = time[i] - time[i-1]
        N[i] = N[i-1] * np.exp(g[i-1] * dt_step)  
    return N, conc

def analytic_population_no_drug(time, N0, birth_rate, death_rate):
    """Closed-form exponential growth/decay with no drug."""
    return N0 * np.exp((birth_rate - death_rate) * time)
