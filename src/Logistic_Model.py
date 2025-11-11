from mesa import Agent, Model
import numpy as np
import matplotlib.pyplot as plt


"""Simple agent-based tumour birth-death model with resource-limited division."""

class TumourCell(Agent):
    """ A single tumour cell agent.

        Attributes:
            model: Reference to the containing TumourModel instance.
            p_birth (float): Per-step probability that this cell attempts to divide.
            p_death (float): Per-step probability that this cell dies.
    """

    def __init__(self, model, p_birth, p_death):
        super().__init__(model)
        self.p_birth = p_birth
        self.p_death = p_death

    def step(self):
        """Perform one agent timestep.

        Behaviour:
        - With probability p_death the cell dies and is removed from the model's
          agent list.
        - If it survives and the model has enough resources (> division_cost),
          it attempts division with probability p_birth. On successful division
          the model's resources are reduced by division_cost and a new
          TumourCell instance is created.

        Notes:
        - Each cell samples birth and death independently.
        - This method mutates model.agents and model.resources directly.
        """

        if np.random.rand() < self.p_death:
            self.model.agents.remove(self)
            return

        if self.model.resources > self.model.division_cost:
            if np.random.rand() < self.p_birth:
                self.model.resources -= self.model.division_cost
                TumourCell(self.model, self.p_birth, self.p_death)


class TumourModel(Model):
    """Container model for tumour population with resource-limited division.

    The model tracks a global resource pool and a collection of TumourCell
    agents. Cells undergo stochastic birth and death each timestep. Continuous
    rates (birth_rate, death_rate) are converted to per-step probabilities
    via p = 1 - exp(-rate * dt).

    Attributes:
        resources (float): Current amount of available resources.
        resource_influx (float): Resources added each timestep.
        division_cost (float): Resource cost of dividing.
        p_birth (float): Per-step division probability derived from birth_rate.
        p_death (float): Per-step death probability derived from death_rate.
        agents (list): Collection of TumourCell instances.
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
    ):
        """Initialise the tumour model.

        Args:
            initial_cells (int): Number of initial tumour cells to create.
            birth_rate (float): Continuous-time birth rate (per unit time).
            death_rate (float): Continuous-time death rate (per unit time).
            dt (float): Timestep duration used to convert rates to probabilities.
            initial_resources (float): Initial amount of available resources.
            resource_influx (float): Amount of resources added each timestep.
            division_cost (float): Resource cost consumed when a cell divides.
        """        
        super().__init__(seed=None)

        self.resources = initial_resources
        self.resource_influx = resource_influx
        self.division_cost = division_cost

        # convert continuous rates to per-step probabilities
        self.p_birth = 1 - np.exp(-birth_rate * dt)
        self.p_death = 1 - np.exp(-death_rate * dt)

        # create initial population
        for i in range(initial_cells):
            TumourCell(self, self.p_birth, self.p_death)

    def step(self):
        """Advance the model state by one timestep.

        Actions performed:
        - Add resource_influx to the global resource pool.
        - Iterate agents and let each perform its step (death/division).
        """

        # Resources
        self.resources += self.resource_influx

        # Agents make independant decisions
        self.agents.shuffle_do("step")


# ------------------- Parameters ------------------- #
timesteps = 300
n_runs = 30  # number of independent simulations
initial_cells = 1

# ------------------- Run Multiple Simulations ------------------- #
all_cell_counts = []
all_resources = []

for run in range(n_runs):
    model = TumourModel(
        initial_cells=initial_cells,
        birth_rate=0.5,
        death_rate=0.05,
        dt=0.1,
        initial_resources=20,
        resource_influx=8,
        division_cost=0.5,
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
timesteps_array = np.arange(timesteps)

# Cell counts
plt.plot(timesteps_array, mean_cells, label="Mean cell count", color="tab:blue")
plt.fill_between(
    timesteps_array,
    mean_cells - std_cells,
    mean_cells + std_cells,
    color="tab:blue",
    alpha=0.2,
)

# Resources
plt.plot(
    timesteps_array,
    mean_resources,
    label="Mean resources",
    color="tab:orange",
    linestyle="--",
)
plt.fill_between(
    timesteps_array,
    mean_resources - std_resources,
    mean_resources + std_resources,
    color="tab:orange",
    alpha=0.2,
)

plt.xlabel("Time step")
plt.ylabel("Count")
plt.title("Mean ± SD over Multiple Runs: Logistic-like Behaviour Emerges")
plt.legend()
plt.tight_layout()
plt.show()
