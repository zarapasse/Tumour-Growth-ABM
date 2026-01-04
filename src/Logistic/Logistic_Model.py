"""
This script gives the basic logistic ABM with logistic fit and parameter values in a panel on the side.
"""

from mesa import Agent, Model
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


"""Agent-based tumour model with explicit per-cell energy bookkeeping.

This module extends a simple birth-death resource-limited model by adding
per-cell energy, metabolic consumption, resource uptake (bounded by a cell's
energy capacity), and energy-dependent division. Key behavioral changes from
a minimal logistic ABM:

- Each TumourCell stores 'energy' and consumes some at each timestep to stay alive. Cells die if energy reaches zero.
- Cells take up resources from the global pool up to their capacity.
- Division requires a minimum energy threshold and sufficient global resources;
  on division a cell halves its energy and spawns a new cell with the same
  energy.
"""


class TumourCell(Agent):
    """A single tumour cell.

    Attributes:
        energy (float): Current internal energy of the cell.
        energy_capacity (float): Maximum energy the cell can store.
    """

    def __init__(self, model, energy=10):
        super().__init__(model)
        self.energy = energy
        self.energy_capacity = model.energy_capacity

    def step(self):
        """
        Perform one cell timestep with energy-based behaviour.

         1. Consume energy (metabolism): if not enough energy in cell, it dies.
         2. Replenish energy by uptaking from the model's global resources.
         3. Attempt division if (a) the cell has sufficient internal energy
            (> 5), (b) the model has at least model.division_cost resources,
            and (c) a random draw exceeds the birth probability.
            On division the model's resources are reduced by
            division_cost and the parent halves its energy; a new TumourCell
            is created with the same halved energy.
         4. Independently of the above, perform a stochastic death draw using
            p_death; on success the cell is removed.

         Notes:
         - All randomness is sampled independently per cell and per timestep.
        """

        self.energy -= self.model.maintenance_cost

        # Death due to energy depletion or stochastic death
        if self.energy <= 0 or np.random.rand() < self.model.p_death:
            self.model.agents.remove(self)
            return

        # Try to take up resources
        uptake = min(self.model.resources, self.energy_capacity - self.energy)
        self.energy += uptake
        self.model.resources -= uptake

        # Division only if enough energy
        if (
            self.energy > self.model.division_threshold
            and self.model.resources >= self.model.division_cost
            and np.random.rand() < self.model.p_birth
        ):
            self.model.resources -= self.model.division_cost
            self.energy /= 2  # split energy with new cell

            TumourCell(self.model, energy=self.energy)


class TumourModel(Model):
    """Container model for tumour population with shared resources.

    This model keeps a global resource pool and a list of TumourCell agents.
    Continuous-time birth and death rates are converted to per-step probabilities
    using p = 1 - exp(-rate * dt).

    """

    def __init__(
        self,
        initial_cells,
        birth_rate,
        death_rate,
        dt,
        initial_resources,
        resource_influx,
        energy_capacity=10,
    ):
        """
        Initialise the tumour model.

        Args:
            initial_cells (int): Number of initial tumour cells to create.
            birth_rate (float): Continuous-time birth rate (per unit time).
            death_rate (float): Continuous-time death rate (per unit time).
            dt (float): Timestep duration used to convert rates to probabilities.
            initial_resources (float): Initial amount of available resources.
            resource_influx (float): Amount of resources added each timestep.
            energy_capacity (float): Maximum energy a cell can store.
        """
        super().__init__(seed=None)

        self.resources = initial_resources
        self.resource_influx = resource_influx
        self.energy_capacity = energy_capacity
        self.division_cost = self.energy_capacity * 0.3
        self.maintenance_cost = self.energy_capacity * 0.02
        self.division_threshold = self.energy_capacity * 0.5

        # convert continuous rates to per-step probabilities
        self.p_birth = 1 - np.exp(-birth_rate * dt)
        self.p_death = 1 - np.exp(-death_rate * dt)

        # create initial population
        for _ in range(initial_cells):
            TumourCell(self)

    def step(self):
        """Advance the model state by one timestep.

        Actions performed:
        - Add resource_influx to the global resource pool.
        - Iterate agents (in random order) and let each perform its step (death/division).
        """

        # Resources
        self.resources += self.resource_influx

        # Agents make independant decisions
        self.agents.shuffle_do("step")


# ------------------- Parameters ------------------- #
timesteps = 500
n_runs = 500  # number of independent simulations
initial_cells = 10
dt = 0.1  # timestep duration
birth_rate = 0.6  # per unit time
death_rate = 0.1  # per unit time
initial_resources = 0
resource_influx = 100
energy_capacity = 10

# ------------------- Run Multiple Simulations ------------------- #
all_cell_counts = []
all_resources = []

for run in range(n_runs):
    model = TumourModel(
        initial_cells=initial_cells,
        birth_rate=birth_rate,
        death_rate=death_rate,
        dt=dt,
        initial_resources=initial_resources,
        resource_influx=resource_influx,
        energy_capacity=energy_capacity,
    )

    cell_counts = []
    resources = []

    for t in range(timesteps):
        model.step()
        cell_counts.append(len(model.agents))
        resources.append(model.resources)

    all_cell_counts.append(cell_counts)
    all_resources.append(resources)

# ------------------- Compute Mean & SD ------------------- #
mean_cells = np.mean(all_cell_counts, axis=0)
std_cells = np.std(all_cell_counts, axis=0)

mean_resources = np.mean(all_resources, axis=0)
std_resources = np.std(all_resources, axis=0)

# ------------------- Plot ------------------- #
plt.figure(figsize=(10, 5))
t = np.arange(timesteps) * dt  # physical time axis

# Cell counts
plt.plot(t, mean_cells, label="Mean cell count", color="tab:blue")
plt.fill_between(
    t,
    mean_cells - std_cells,
    mean_cells + std_cells,
    color="tab:blue",
    alpha=0.2,
)

# ------------------- Logistic fit ------------------- #

def logistic_function(t, K, r, N0):
    """Standard logistic growth equation.
    Args:
        t: time array
        K: carrying capacity
        r: growth rate
        N0: initial population
    """
    return K / (1 + ((K - N0) / N0) * np.exp(-r * t))


def fit_logistic_direct(t, N):
    """Direct nonlinear curve fitting.

    This directly fits the logistic equation to the data using
    scipy's curve_fit optimizer.
    """
    # Initial parameter guesses
    K_guess = np.max(N) * 1.1  # slightly above max
    N0_guess = N[0]
    r_guess = 0.1

    try:
        # Fit the logistic function
        params, _ = curve_fit(
            logistic_function,
            t,
            N,
            p0=[K_guess, r_guess, N0_guess],
            bounds=([0, 0, 0], [np.inf, np.inf, np.inf]),
            maxfev=10000,
        )
        K, r, N0_fit = params
        return K, r, N0_fit, logistic_function(t, K, r, N0_fit)
    except:
        return None, None, None, None

# ------------------- Plot ------------------- #
fig, ax = plt.subplots(figsize=(10, 6))
t = np.arange(timesteps) * dt

# Cell counts
ax.plot(t, mean_cells, lw=2, label="Mean cell count")
ax.fill_between(
    t,
    mean_cells - std_cells,
    mean_cells + std_cells,
    alpha=0.25,
    label="Mean ± SD",
)

# ------------------- Logistic fit ------------------- #
K1, r1, N0_1, fit1 = fit_logistic_direct(t, mean_cells)
ax.plot(t, fit1, "k--", lw=2, label=f"Logistic fit")

ax.set_xlabel("Time")
ax.set_ylabel("Count")
ax.set_title("Resource-dependent ABM vs Logistic Fit (no drug)")
ax.legend()
ax.grid(alpha=0.3)

# ------------------- Parameter panel (same layout style) ------------------- #

param_lines = [
    "Simulation parameters",
    "----------------------",
    f"timesteps        = {timesteps}",
    f"dt              = {dt}",
    f"n_runs          = {n_runs}",
    f"initial_cells   = {initial_cells}",
    "",
    "Model parameters",
    "----------------------",
    f"birth_rate       = {birth_rate}",
    f"death_rate       = {death_rate}",
    f"initial_resources= {initial_resources}",
    f"resource_influx  = {resource_influx}",
    f"energy_capacity  = {energy_capacity}",
    "",
    "Logistic fit (mean)",
    "----------------------",
    f"K   = {K1:.3g}",
    f"r   = {r1:.3g}",
    f"N0  = {N0_1:.3g}",
]

param_text = "\n".join(param_lines)

# Leave space on the right for the panel 
plt.tight_layout(rect=(0, 0, 0.82, 1.0))

fig.text(
    0.84,
    0.95,
    param_text,
    va="top",
    ha="left",
    fontsize=8,
    family="monospace",
    bbox=dict(
        boxstyle="round",
        facecolor="white",
        edgecolor="0.8",
        alpha=0.95,
    ),
)

plt.tight_layout(rect=(0, 0, 0.82, 1.0))
plt.savefig("logistic_fit_no_drug.png", dpi=300, bbox_inches="tight")
plt.show()

