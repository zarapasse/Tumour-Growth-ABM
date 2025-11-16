import numpy as np
from Exponential_Model import TumourModel



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



#---------------------- ABM simulation runner ---------------------- #
def run_abm_for_schedule(config, schedule, alpha_val):
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
