from mesa import Agent, Model
import numpy as np
import matplotlib.pyplot as plt


"""
Agent-based birth–death tumour model with optional drug effect.

A compact stochastic ABM where each tumour cell samples independent birth and
death events each timestep. The model supports an external, discrete drug
schedule that instantaneously sets a drug concentration for specified
intervals. Drug effect on proliferation is modelled with a Hill-type
inhibition function.

Flexible bolus drug scheduling is allowed, and PK effects have been added.

"""


class TumourCell(Agent):
    """Single tumour cell agent.

    Behaviour:
    - On each step the cell samples independent Bernoulli events based on
      the model's current p_birth and p_death.
    - If a birth event occurs the cell divides.
    - If a death event occurs the cell is removed.
    """

    def __init__(self, model):
        super().__init__(model)
        self.p_birth = model.p_birth
        self.p_death = model.p_death

    def step(self):
        """Sample birth and death events for each cell independently to determine cell behaviour."""
        if np.random.rand() < self.model.p_birth:
            TumourCell(self.model)
        if np.random.rand() < self.model.p_death:
            self.model.agents.remove(self)


class TumourModel(Model):
    """
    Responsibilities:
    - Convert continuous birth_rate/death_rate to per-step probabilities using
      p = 1 - exp(-rate * dt).
    - Maintain an optional drug_schedule (list of (start_time, duration, conc))
    - Compute drug inhibition using a Hill function
      and reduce the effective death rate multiplicatively by Hill_multiplier.
    - Advance time and run agent steps in randomised order.


    Attributes:
        birth_rate (float): continuous-time birth rate (per unit time).
        death_rate (float): continuous-time death rate (per unit time).
        dt (float): timestep size used to convert continuous rates to per-step probabilities.
        p_birth (float): current per-step birth probability (updated when drug is present).
        p_death (float): per-step death probability (derived from death_rate).
        drug_schedule (list): list of (amount, time) tuples for bolus dosing.
        drug_conc (float): current drug concentration.
        time (float): model time (advances by dt on each step).
        E0, E1, C, n: parameters for the Hill function.
        alpha (float): drug decay rate (PK).

    """

    def __init__(
        self, initial_cells, birth_rate, death_rate, dt, drug_schedule=None, alpha=0.5
    ):
        super().__init__(seed=None)

        self.birth_rate = birth_rate
        self.death_rate = death_rate
        self.dt = dt
        self.p_birth = 1 - np.exp(-birth_rate * dt)
        self.p_death = 1 - np.exp(-death_rate * dt)

        # create initial population
        for i in range(initial_cells):
            TumourCell(self)

        # Drug parameters
        self.drug_schedule = drug_schedule if drug_schedule is not None else []
        self.drug_conc = 0.0
        self.time = 0.0
        self.alpha = alpha
        self.administered_doses = []

        self.E0 = 1.0  # Baseline multiplier for death rate (no drug effect)
        self.E1 = (
            2.0  # Maximum multiplier for death rate under drug (e.g. up to 3x increase)
        )
        self.C = 1.0  # EC50 - concentration for half-maximal effect
        self.n = 2.0  # Hill coefficient

    def pk_dynamics(self, current_time):
        """
        Calculate total drug concentration from all administered doses.

        For each dose in administered_doses, calculates:
        concentration = amount * exp(-alpha * (current_time - administration_time))

        Then sums all active doses to get total concentration.

        Args:
            current_time (float): current simulation time

        Returns:
            float: total drug concentration from all active doses
        """
        total_conc = 0.0

        active_doses = []
        for amount, dose_time in self.administered_doses:
            time_since_dose = current_time - dose_time
            if time_since_dose >= 0:
                dose_conc = amount * np.exp(-self.alpha * time_since_dose)
                total_conc += dose_conc

                if dose_conc > 0:
                    active_doses.append((amount, dose_time))

        self.administered_doses = active_doses
        return total_conc

    def check_drug_schedule(self, current_time):

        dose_given = []
        for i, (amount, dose_time) in enumerate(self.drug_schedule):
            if dose_time <= current_time < dose_time + self.dt:
                self.administered_doses.append((amount, dose_time))
                dose_given.append(i)
                print(f"Administered dose: {amount} at time {current_time:.2f}")

        for i in sorted(dose_given, reverse=True):
            self.drug_schedule.pop(i)

    def hill_equation(self, drug_conc=0.0):
        """Calculate death-rate multiplier using Hill equation:
        H(x) = E0 + (x^n (E1 - E0)) / (x^n + C^n)
        Returns a multiplier (>= E0) that should be applied to the baseline death rate.
        E1-E0: maximum increase in death rate due to drug.
        """
        if drug_conc <= 0:
            return self.E0

        Hill_multiplier = self.E0 + (drug_conc**self.n * (self.E1 - self.E0)) / (
            drug_conc**self.n + self.C**self.n
        )

        return Hill_multiplier

    def update_drug_concentration(self):
        """Set drug_conc according to the schedule, then advance model time by dt."""
        self.check_drug_schedule(self.time)
        self.drug_conc = self.pk_dynamics(self.time)

        self.time += self.dt

    def step(self):
        """Advance the model by one timestep.

        Workflow:
        1. Update the drug concentration and model time.
        2. If drug is present, compute death-rate multiplier via hill_equation and increase
           the effective death rate: effective_death_rate = death_rate * Hill_multiplier.
           Update self.p_death accordingly.
        3. Otherwise restore p_death from the baseline death_rate.
        4. Always keep p_birth at baseline (drug affects death only).
        """

        self.update_drug_concentration()
        current_drug_conc = self.drug_conc

        # baseline p_birth
        self.p_birth = 1 - np.exp(-self.birth_rate * self.dt)

        if current_drug_conc > 0:
            Hill_multiplier = self.hill_equation(current_drug_conc)
            effective_death_rate = self.death_rate * Hill_multiplier
            self.p_death = 1 - np.exp(-effective_death_rate * self.dt)
        else:
            self.p_death = 1 - np.exp(-self.death_rate * self.dt)

        self.agents.shuffle_do("step")


# -------------- Simulation and Plotting Code --------------
# Define Parameters
initial_cells = 1000
birth_rate = 0.2
death_rate = 0.1
dt = 0.1
steps = 200  # Increased to see more dynamics
T = steps * dt
n_runs = 10  # Reduced for faster testing

# Drug schedule: list of (amount, time) for bolus dosing
drug_schedule = [
    (10.0, 0.0),  
    (10.0, 10.0),  
]

# Run simulations with and without drug for comparison
all_populations_no_drug = np.zeros((n_runs, steps))
all_populations_with_drug = np.zeros((n_runs, steps))

# Also track drug concentration for one run
drug_concentrations = []

print("Running simulations without drug...")
for run in range(n_runs):
    model = TumourModel(initial_cells, birth_rate, death_rate, dt)
    population_sizes = []
    for step in range(steps):
        model.step()
        population_sizes.append(len(model.agents))
    all_populations_no_drug[run, :] = population_sizes

print("Running simulations with drug...")
for run in range(n_runs):
    model = TumourModel(
        initial_cells, birth_rate, death_rate, dt, drug_schedule=drug_schedule.copy()
    )
    population_sizes = []
    concentration_history = []
    for step in range(steps):
        model.step()
        population_sizes.append(len(model.agents))
        concentration_history.append(model.drug_conc)

    all_populations_with_drug[run, :] = population_sizes

    # Store drug concentrations from first run
    if run == 0:
        drug_concentrations = concentration_history

# Compute statistics
mean_no_drug = all_populations_no_drug.mean(axis=0)
std_no_drug = all_populations_no_drug.std(axis=0)
mean_with_drug = all_populations_with_drug.mean(axis=0)
std_with_drug = all_populations_with_drug.std(axis=0)

# Time array for plotting
time = np.arange(steps) * dt

# ========== PLOTTING ==========
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Plot 1: Tumour cell populations
ax1.plot(time, mean_no_drug, "b-", label="No Drug - ABM Mean", linewidth=2)
ax1.fill_between(
    time,
    mean_no_drug - std_no_drug,
    mean_no_drug + std_no_drug,
    color="blue",
    alpha=0.2,
    label="No Drug - ±1 Std Dev",
)
ax1.plot(time, mean_with_drug, "r-", label="With Drug - ABM Mean", linewidth=2)
ax1.fill_between(
    time,
    mean_with_drug - std_with_drug,
    mean_with_drug + std_with_drug,
    color="red",
    alpha=0.2,
    label="With Drug - ±1 Std Dev",
)

# Mark drug administration times
for amount, dose_time in drug_schedule:
    ax1.axvline(x=dose_time, color="gray", linestyle="--", alpha=0.7)
    ax1.text(
        dose_time,
        ax1.get_ylim()[1] * 0.95,
        f"Dose: {amount}",
        rotation=90,
        va="top",
        ha="right",
        fontsize=8,
    )

ax1.set_ylabel("Number of cells")
ax1.set_title("Tumour Cell Population Dynamics with Bolus Dosing")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Drug concentration
ax2.plot(time, drug_concentrations, "g-", label="Drug Concentration", linewidth=2)

# Mark drug administration times
for amount, dose_time in drug_schedule:
    ax2.axvline(x=dose_time, color="gray", linestyle="--", alpha=0.7)
    ax2.text(
        dose_time,
        ax2.get_ylim()[1] * 0.9,
        f"Dose: {amount}",
        rotation=90,
        va="top",
        ha="right",
        fontsize=8,
    )

ax2.set_xlabel("Time")
ax2.set_ylabel("Drug Concentration")
ax2.set_title("PK Drug Concentration Profile")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print diagnostic information
print(f"\nModel Parameters:")
print(f"Initial cells: {initial_cells}")
print(f"Birth rate: {birth_rate}, Death rate: {death_rate}")
print(f"PK decay rate (alpha): {0.1}")
print(f"Growth rate without drug: {birth_rate - death_rate:.3f}")
print(
    f"Expected doubling time without drug: {np.log(2)/(birth_rate - death_rate):.2f} time units"
)

# Final populations
final_abm_no_drug = mean_no_drug[-1]
final_abm_with_drug = mean_with_drug[-1]

print(f"\nFinal populations:")
print(f"No Drug: {final_abm_no_drug:.0f} cells")
print(f"With Drug: {final_abm_with_drug:.0f} cells")
print(f"Tumour suppression: {(1 - final_abm_with_drug/final_abm_no_drug)*100:.1f}%")

# Show dosing schedule
print(f"\nDosing schedule:")
for i, (amount, time) in enumerate(drug_schedule):
    print(f"  Dose {i+1}: {amount} at time {time}")
