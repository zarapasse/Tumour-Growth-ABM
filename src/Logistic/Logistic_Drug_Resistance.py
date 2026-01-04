from mesa import Agent, Model
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from mesa.datacollection import DataCollector


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
        self.cell_type = "sensitive"

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
            
            if np.random.rand() < self.model.p_mutation:
                TumourResistantCell(self.model, energy=self.energy)
            else:
                TumourCell(self.model, energy=self.energy)

class TumourResistantCell(Agent):
    """A resistant tumour cell that is unaffected by the drug.
    """

    def __init__(self, model, energy=10):
        super().__init__(model)
        self.energy = energy
        self.energy_capacity = model.energy_capacity
        self.cell_type = "resistant"
        self.p_death = 1 - np.exp(-model.death_rate * model.dt)  # baseline death only (no drug effect)

    def step(self):
        """
        Perform one cell timestep with energy-based behaviour.

         1. Consume energy (metabolism): if not enough energy in cell, it dies.
         2. Replenish energy by uptaking from the model's global resources.
         3. Attempt division if (a) the cell has sufficient internal energy
            (> 5), (b) the model has at least model.division_cost resources,
            and (c) a random draw exceeds the birth probability.
            On division the model's resources are reduced by
            division_cost and the parent halves its energy; a new TumourResistantCell
            is created with the same halved energy.
         4. Independently of the above, perform a stochastic death draw using
            p_death; on success the cell is removed.

         Notes:
         - All randomness is sampled independently per cell and per timestep.
        """

        self.energy -= self.model.maintenance_cost

        # Death due to energy depletion or stochastic death
        if self.energy <= 0 or np.random.rand() < self.p_death:
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

            TumourResistantCell(self.model, energy=self.energy)


class TumourResistantModel(Model):
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
        alpha,
        p_mutation,
        hill_params=None,
        drug_schedule=None,
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
            alpha (float): first-order PK decay rate for the drug.
            drug_schedule (list or None): Schedule of drug doses as a list of (time, dose) tuples.
            hill_params (dict or None): Parameters for the Hill function (e.g., {"n": value, "C": value}).
        """
        super().__init__(seed=None)

        self.resources = initial_resources
        self.resource_influx = resource_influx
        self.energy_capacity = energy_capacity
        self.division_cost = self.energy_capacity * 0.1
        self.maintenance_cost = self.energy_capacity * 0.01
        self.division_threshold = self.energy_capacity * 0.5

        self.birth_rate = birth_rate
        self.death_rate = death_rate
        self.dt = dt
        self.p_mutation = p_mutation

        # convert continuous rates to per-step probabilities
        self.p_birth = 1 - np.exp(-birth_rate * dt)
        self.p_death = 1 - np.exp(-death_rate * dt)

        # create initial population
        for _ in range(initial_cells):
            TumourCell(self)

        # Drug parameters
        self.drug_schedule = drug_schedule if drug_schedule is not None else []
        self.drug_conc = 0.0
        self.time = 0.0
        self.alpha = alpha
        if hill_params is None:
            self.E0, self.E1, self.C, self.n = 0.0, 0.0, 1.0, 1.0  # No drug effect
        else:
            self.E0, self.E1, self.C, self.n = (
                hill_params["E0"],
                hill_params["E1"],
                hill_params["C"],
                hill_params["n"],
            )
            
        #Mesa DataCollector to track populations
        self.datacollector = DataCollector(
            model_reporters={
                "Sensitive": lambda m: sum(
                    1 for a in m.agents if type(a) is TumourCell
                ),
                "Resistant": lambda m: sum(
                    1 for a in m.agents if isinstance(a, TumourResistantCell)
                ),
                "Total": lambda m: len(m.agents),
                "ResistantFraction": lambda m: (
                    sum(1 for a in m.agents if isinstance(a, TumourResistantCell))
                    / max(1, len(m.agents))
                ),
                "DrugConc": lambda m: m.drug_conc,
            }
        )        
        
        self.datacollector.collect(self)

    def pk_dynamics(self, current_time):
        """
        Compute total drug concentration driectly using exact PK decay at current time.
        """
        total_conc = 0.0
        for amount, dose_time in self.drug_schedule:
            if current_time >= dose_time:
                total_conc += amount * np.exp(-self.alpha * (current_time - dose_time))
        return total_conc

    def hill_equation(self, drug_conc=0.0):
        """Calculate drug-induced cell death using the Hill equation:
        H(x) = E0 + (x^n (E1 - E0)) / (x^n + C^n)
        """
        if drug_conc <= 0:
            return self.E0

        kill_rate = self.E0 + (drug_conc**self.n * (self.E1 - self.E0)) / (
            drug_conc**self.n + self.C**self.n
        )
        return kill_rate

    def update_drug_concentration(self):
        """Update drug concentration based on dosing schedule and PK dynamics."""
        self.drug_conc = self.pk_dynamics(self.time)
        self.time += self.dt

    def step(self):
        """Advance the model state by one timestep.

        Actions performed:
        - Add resource_influx to the global resource pool.
        - Update drug concentration based on dosing schedule and PK dynamics.
        - Iterate agents (in random order) and let each perform its step (death/division).
        """
        # Update drug concentration
        current_drug_conc = self.drug_conc
        self.update_drug_concentration()

        if current_drug_conc > 0:
            kill_rate = self.hill_equation(current_drug_conc)
            effective_death_rate = self.death_rate + kill_rate
            self.p_death = 1 - np.exp(-effective_death_rate * self.dt)
        else:
            self.p_death = 1 - np.exp(-self.death_rate * self.dt)

        # Resources
        self.resources += self.resource_influx

        # Agents make independant decisions
        self.agents.shuffle_do("step")
        self.datacollector.collect(self)


# # ------------------- Parameters ------------------- #
# timesteps = 500
# n_runs = 1000  # number of independent simulations
# initial_cells = 65
# dt = 0.1  # timestep duration

# #Example drug parameters
# alpha = 1
# drug_schedule = [(50.0, 5.0)]
# hill_params = {"E0": 0.00, "E1": 1.0, "C": 35.0, "n": 5.0}

# # ------------------- Run Multiple Simulations ------------------- #
# all_cell_counts = []
# all_resources = []

# for run in range(n_runs):
#     model = TumourModel(
#         initial_cells=initial_cells,
#         birth_rate=0.2,
#         death_rate=0.1,
#         dt=dt,
#         initial_resources=100,
#         resource_influx=10,
#         energy_capacity=10,
#         alpha=alpha,
#         p_mutation=0.01,
#         drug_schedule=drug_schedule,
#         hill_params=hill_params,
#     )

#     cell_counts = []
#     resources = []

#     for t in range(timesteps):
#         model.step()
#         cell_counts.append(len(model.agents))
#         resources.append(model.resources)

#     all_cell_counts.append(cell_counts)
#     all_resources.append(resources)

# # ------------------- Compute Mean & SD ------------------- #
# mean_cells = np.mean(all_cell_counts, axis=0)
# std_cells = np.std(all_cell_counts, axis=0)

# mean_resources = np.mean(all_resources, axis=0)
# std_resources = np.std(all_resources, axis=0)

# # ------------------- Plot ------------------- #
# plt.figure(figsize=(10, 5))
# t = np.arange(timesteps) * dt  # physical time axis

# # Cell counts
# plt.plot(t, mean_cells, label="Mean cell count", color="tab:blue")
# plt.fill_between(
#     t,
#     mean_cells - std_cells,
#     mean_cells + std_cells,
#     color="tab:blue",
#     alpha=0.2,
# )

# # ------------------- Logistic fit ------------------- #


# def logistic_function(t, K, r, N0):
#     """Standard logistic growth equation.
#     Args:
#         t: time array
#         K: carrying capacity
#         r: growth rate
#         N0: initial population
#     """
#     return K / (1 + ((K - N0) / N0) * np.exp(-r * t))


# def fit_logistic_direct(t, N):
#     """Direct nonlinear curve fitting.

#     This directly fits the logistic equation to the data using
#     scipy's curve_fit optimizer.
#     """
#     # Initial parameter guesses
#     K_guess = np.max(N) * 1.1  # slightly above max
#     N0_guess = N[0]
#     r_guess = 0.1

#     try:
#         # Fit the logistic function
#         params, _ = curve_fit(
#             logistic_function,
#             t,
#             N,
#             p0=[K_guess, r_guess, N0_guess],
#             bounds=([0, 0, 0], [np.inf, np.inf, np.inf]),
#             maxfev=10000,
#         )
#         K, r, N0_fit = params
#         return K, r, N0_fit, logistic_function(t, K, r, N0_fit)
#     except:
#         return None, None, None, None


# K1, r1, N0_1, fit1 = fit_logistic_direct(t, mean_cells)

# plt.plot(t, fit1, "k--", label=f"Logistic Fit: K={K1:.1f}, r={r1:.3f}")

# plt.xlabel("Time")
# plt.ylabel("Count")
# plt.title("Mean ± SD over Multiple Runs with Logistic Fit")
# plt.legend()
# plt.tight_layout()
# plt.show()
