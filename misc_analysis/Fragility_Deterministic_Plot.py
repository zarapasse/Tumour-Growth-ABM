import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


# ---------------------- Local HillParams ---------------------- #
@dataclass
class HillParams:
    K_kill: float
    C: float
    n: float


# ---------------------- Copied from utils ---------------------- #
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

    p_mutation = sim.get("p_mutation", 0.0)
    initial_resistant_fraction = sim.get("initial_resistant_fraction", 0.0)
    fitness_cost = sim.get("fitness_cost", 0.0)

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
    """
    mean_dose = total_dose_per_cycle / n_doses_per_cycle
    dt_dose = cycle_length / n_doses_per_cycle

    even_schedule = []
    for c in range(n_cycles):
        cycle_start = c * cycle_length
        for i in range(n_doses_per_cycle):
            even_schedule.append((mean_dose, cycle_start + i * dt_dose))

    deviations = [sigma if i % 2 == 0 else -sigma for i in range(n_doses_per_cycle)]
    if n_doses_per_cycle % 2 == 1:
        deviations[-1] = 0.0

    odd_schedule = []
    for c in range(n_cycles):
        cycle_start = c * cycle_length
        for i, dev in enumerate(deviations):
            odd_schedule.append((mean_dose + dev, cycle_start + i * dt_dose))

    return [
        {"name": "Even Schedule", "schedule": even_schedule, "alpha": alpha},
        {"name": "Uneven Schedule", "schedule": odd_schedule, "alpha": alpha},
    ]


def hill_kill_rate(conc, hill_params):
    """
    Saturating Hill kill term:
        k_kill(c) = K_kill * c^n / (c^n + C^n)
    """
    K_kill = float(hill_params.K_kill)
    C = float(hill_params.C)
    n = float(hill_params.n)

    c_n = conc**n
    denominator = c_n + (C**n)
    return K_kill * (c_n / denominator)


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


# ---------------------- Load config ---------------------- #
CONFIG_PATH = (
    Path(__file__).resolve().parent.parent / "src" / "Exponential" / "config.json"
)

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

initial_cells, birth_rate, death_rate, dt, steps, n_runs, hill_params, _, _, _ = (
    process_config(config)
)


# ---------------------- Settings ---------------------- #
n_doses_per_cycle = 2
n_cycles = 1
cycle_length = 12.0
alpha = 1.0

t_end = n_cycles * cycle_length
time = np.arange(0.0, t_end + dt, dt)

x_bar_values = np.arange(10, 100, 2.5)
fragility = []


# ---------------------- Main sweep ---------------------- #
for x_bar in x_bar_values:
    total_dose_per_cycle = x_bar * n_doses_per_cycle

    # For 2 doses/cycle, sigma = x_bar gives (2*x_bar, 0)
    sigma = x_bar

    scenarios = make_fragility_test_scenarios(
        total_dose_per_cycle=total_dose_per_cycle,
        n_doses_per_cycle=n_doses_per_cycle,
        n_cycles=n_cycles,
        sigma=sigma,
        cycle_length=cycle_length,
        alpha=alpha,
    )

    even_sc, odd_sc = scenarios[0], scenarios[1]

    N_even, _ = analytic_population_with_pk(
        time=time,
        N0=initial_cells,
        birth_rate=birth_rate,
        death_rate=death_rate,
        schedule=even_sc["schedule"],
        alpha=even_sc["alpha"],
        hill_params=hill_params,
    )

    N_odd, _ = analytic_population_with_pk(
        time=time,
        N0=initial_cells,
        birth_rate=birth_rate,
        death_rate=death_rate,
        schedule=odd_sc["schedule"],
        alpha=odd_sc["alpha"],
        hill_params=hill_params,
    )

    F = (N_odd[-1] - N_even[-1]) / initial_cells
    fragility.append(F)

fragility = np.array(fragility)


# ---------------------- Plot ---------------------- #
fig, ax = plt.subplots(figsize=(8.8, 5.4))

ax.plot(x_bar_values, fragility, color="black", linewidth=2.8, zorder=3)
ax.axhline(0, color="gray", linestyle="--", linewidth=1.2, zorder=2)

ax.fill_between(
    x_bar_values,
    fragility,
    0,
    where=fragility >= 0,
    alpha=0.18,
    interpolate=True,
    zorder=1,
)

ax.fill_between(
    x_bar_values,
    fragility,
    0,
    where=fragility <= 0,
    alpha=0.10,
    interpolate=True,
    zorder=1,
)

ax.text(
    0.52,
    0.92,
    "Even dosing preferred",
    transform=ax.transAxes,
    ha="center",
    va="center",
    fontsize=13,
    fontweight="bold",
    alpha=0.8,
)

ax.text(
    0.52,
    0.14,
    "Uneven dosing preferred",
    transform=ax.transAxes,
    ha="center",
    va="center",
    fontsize=13,
    fontweight="bold",
    alpha=0.8,
)

ax.text(0.02, 0.97, r"$F>0$", transform=ax.transAxes, fontsize=11, va="top")
ax.text(0.02, 0.03, r"$F<0$", transform=ax.transAxes, fontsize=11, va="bottom")

ax.set_xlabel(r"Mean dose $\bar{x}$", fontsize=13)
ax.set_ylabel(r"Fragility $F$", fontsize=13)

ax.grid(True, alpha=0.2)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()

outdir = Path(__file__).parent / "graphs"
outdir.mkdir(parents=True, exist_ok=True)
outpath = outdir / "Fragility_Deterministic.png"

plt.savefig(outpath, dpi=300, bbox_inches="tight")
plt.show()
print(f"Saved plot to: {outpath}")
