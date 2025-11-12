from mesa import Agent, Model
import numpy as np
import matplotlib.pyplot as plt


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
        self.energy_capacity = 10

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

        self.energy -= self.model.maintainance_cost

        if self.energy <= 0:
            self.model.agents.remove(self)
            return

        # Try to take up resources
        uptake = min(self.model.resources, self.energy_capacity - self.energy)
        self.energy += uptake
        self.model.resources -= uptake

        # Division only if enough energy
        if (
            self.energy > 5
            and self.model.resources >= self.model.division_cost
            and np.random.rand() < self.model.p_birth
        ):
            self.model.resources -= self.model.division_cost
            self.energy /= 2  # split energy with new cell

            TumourCell(self.model, energy=self.energy)

        # Natural death
        if np.random.rand() < self.model.p_death:
            self.model.agents.remove(self)
            return


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
        division_cost,
        maintainance_cost,
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
            division_cost (float): Resource cost consumed when a cell divides.
            maintainance_cost (float): Cost per cell per timestep.
        """
        super().__init__(seed=None)

        self.resources = initial_resources
        self.resource_influx = resource_influx
        self.division_cost = division_cost
        self.maintainance_cost = maintainance_cost

        # convert continuous rates to per-step probabilities
        self.p_birth = 1 - np.exp(-birth_rate * dt)
        self.p_death = 1 - np.exp(-death_rate * dt)

        # create initial population
        for i in range(initial_cells):
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
n_runs = 50  # number of independent simulations
initial_cells = 1
dt = 0.1  # timestep duration

# ------------------- Run Multiple Simulations ------------------- #
all_cell_counts = []
all_resources = []

for run in range(n_runs):
    model = TumourModel(
        initial_cells=initial_cells,
        birth_rate=0.7,
        death_rate=0.01,
        dt=dt,
        initial_resources=100,
        resource_influx=10,
        division_cost=1,
        maintainance_cost=0.1,
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

# ------------------- Logistic fit (per-capita growth regression) ------------------- #
eps = 1e-9
N = mean_cells.astype(float)
N_safe = np.maximum(N, eps)

# Instantaneous per-capita growth over dt
r_inst = (1.0 / dt) * np.log(N_safe[1:] / N_safe[:-1])
N_mid = N[:-1]
t_mid = t[:-1]

# Robust initial K from late segment
late = max(5, int(0.2 * len(N_mid)))
K_init = max(np.median(N_mid[-late:]), np.max(N_mid) * 0.9)

# Use mid-range data for linear fit: r(N) = r - (r/K) N
mid_mask = (N_mid > 0.1 * K_init) & (N_mid < 0.9 * K_init) & np.isfinite(r_inst)
if np.count_nonzero(mid_mask) >= 5:
    slope, intercept = np.polyfit(N_mid[mid_mask], r_inst[mid_mask], 1)
    # r(N) = intercept + slope * N  => r_hat = intercept, K_hat = -intercept / slope
    r_hat = float(intercept)
    K_hat = float(-intercept / slope) if slope < 0 else np.nan

    if np.isfinite(K_hat) and K_hat > 0 and r_hat > 0:
        N0 = max(N[0], eps)
        logistic = K_hat / (1.0 + ((K_hat - N0) / N0) * np.exp(-r_hat * t))
        plt.plot(
            t,
            logistic,
            "k--",
            linewidth=2,
            alpha=0.9,
            label=f"Logistic fit (K={K_hat:.1f}, r={r_hat:.3f})",
        )
    else:
        print("Warning: logistic parameters not identifiable (bad slope). Skipping.")
else:
    print("Warning: not enough mid-range points to fit logistic curve; skipping.")

plt.xlabel("Time")
plt.ylabel("Count")
plt.title("Mean ± SD over Multiple Runs with Logistic Fit")
plt.legend()
plt.tight_layout()
plt.show()
