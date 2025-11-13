from mesa import Agent, Model
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

# Load parameters from JSON file
CONFIG_PATH = Path(__file__).parent / "config.json"
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

sim_params = config["simulation"]
drug_params = config["drug"]
hill_params = config["hill_parameters"]


"""
Agent-based birth–death tumour model with optional drug effect.

Pharmacology:
- Drug is administered as bolus doses at specific times and follows
  first-order decay with rate alpha (PK).
- Drug action increases death rate: the effective death rate is death_rate + kill_rate(Conc), 
  where kill_rate(Conc) is a Hill function of the instantaneous concentration Conc(t).
- The birth rate is unaffected by the drug.
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
        if np.random.rand() < self.model.p_birth:
            TumourCell(self.model)
        if np.random.rand() < self.model.p_death:
            self.model.agents.remove(self)


class TumourModel(Model):
    """
    Responsibilities:
    - Convert continuous birth_rate/death_rate to per-step probabilities using
      p = 1 - exp(-rate * dt).
    - Track and update drug concentration based on bolus dosing schedule and
      first-order decay.
    - Compute drug induced kill via a Hill function
    - Advance time and run agent steps in randomised order.

    Attributes:
        birth_rate (float): continuous-time birth rate (per unit time).
        death_rate (float): continuous-time baseline death rate (per unit time).
        dt (float): timestep size used to convert continuous rates to per-step probabilities.
        p_birth (float): current per-step birth probability (drug does not affect birth).
        p_death (float): current per-step death probability (from effective death rate).
        drug_schedule (list[tuple[float, float]]): bolus doses as (amount, time).
        drug_conc (float): current drug concentration.
        time (float): simulation time (advances by dt each step).
        E0, E1, C, n: Hill parameters for kill_rate(C):
            - E0: baseline kill rate at zero concentration.
            - E1: maximal kill rate at saturating concentration.
            - C: EC50 (half-maximal concentration).
            - n: Hill coefficient (steepness).
        alpha (float): first-order PK decay rate for the drug.
    """

    def __init__(
        self,
        initial_cells,
        birth_rate,
        death_rate,
        dt,
        drug_schedule=None,
        alpha=0.5,
        hill_params=None,
    ):
        super().__init__(seed=None)

        self.birth_rate = birth_rate
        self.death_rate = death_rate
        self.dt = dt
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
        self.administered_doses = []

        self.E0, self.E1, self.C, self.n = (
            hill_params["E0"],
            hill_params["E1"],
            hill_params["C"],
            hill_params["n"],
        )

    def pk_dynamics(self, current_time):
        """
        Compute total drug concentration as the sum of exponentials from all
        administered bolus doses with first-order decay
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
        self.check_drug_schedule(self.time)
        self.drug_conc = self.pk_dynamics(self.time)

        self.time += self.dt

    def step(self):
        """Advance the model by one timestep.

        Workflow:
        1. Update the drug concentration and model time.
        2. If drug is present, drug induced death via hill_equation and increase
           the effective death rate: effective_death_rate = death_rate + kill_rate.
           Update self.p_death accordingly.
        3. Otherwise restore p_death from the baseline death_rate.
        4. Always keep p_birth at baseline (drug affects death only).
        """

        self.update_drug_concentration()
        current_drug_conc = self.drug_conc

        # baseline p_birth
        self.p_birth = 1 - np.exp(-self.birth_rate * self.dt)

        if current_drug_conc > 0:
            kill_rate = self.hill_equation(current_drug_conc)
            effective_death_rate = self.death_rate + kill_rate
            self.p_death = 1 - np.exp(-effective_death_rate * self.dt)
        else:
            self.p_death = 1 - np.exp(-self.death_rate * self.dt)

        self.agents.shuffle_do("step")

# Analytical deterministic model for comparison
def hill_effect(x, n, C, E0=0.0, E1=0.5):
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
        N[i] = N[i-1] * np.exp(g[i-1] * dt_step)  # left-Riemann, consistent with ABM stepping
    return N, conc

def analytic_population_no_drug(time, N0, birth_rate, death_rate):
    """Closed-form exponential growth/decay with no drug."""
    return N0 * np.exp((birth_rate - death_rate) * time)


















# -------------- Simulation and Plotting Code --------------

# Define Parameters
initial_cells = sim_params["initial_cells"]
birth_rate = sim_params["birth_rate"]
death_rate = sim_params["death_rate"]
dt = sim_params["dt"]
steps = sim_params["steps"]
T = steps * dt
n_runs = sim_params["n_runs"]

# Drug schedule: list of (amount, time) for bolus dosing
drug_schedule = [tuple(d) for d in drug_params["schedule"]]  # list of (amount, time)
alpha = drug_params["alpha"]

# Run simulations with and without drug for comparison
all_populations_no_drug = np.zeros((n_runs, steps))
all_populations_with_drug = np.zeros((n_runs, steps))

# Also track drug concentration for one run
drug_concentrations = []

print("Running simulations without drug...")
for run in range(n_runs):
    model = TumourModel(
        initial_cells,
        birth_rate,
        death_rate,
        dt,
        drug_schedule=[],
        alpha=alpha,
        hill_params=hill_params,
    )
    population_sizes = []
    for step in range(steps):
        model.step()
        population_sizes.append(len(model.agents))
    all_populations_no_drug[run, :] = population_sizes

print("Running simulations with drug...")
for run in range(n_runs):
    model = TumourModel(
        initial_cells,
        birth_rate,
        death_rate,
        dt,
        drug_schedule=drug_schedule.copy(),
        alpha=alpha,
        hill_params=hill_params,
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


# -------- Deterministic (config-driven) curves for overlay --------
N_det, conc_det = analytic_population_with_pk(
    time, initial_cells, birth_rate, death_rate, drug_schedule, alpha, hill_params
)
N_no_drug_det = analytic_population_no_drug(time, initial_cells, birth_rate, death_rate)

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
# Overlays
ax1.plot(time, N_no_drug_det, "b--", linewidth=2, alpha=0.85, label="Analytic (no drug)")
ax1.plot(time, N_det, "k--", linewidth=2, label="Deterministic (with PK + Hill)")

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
ax2.plot(time, drug_concentrations, "g-", label="Drug Concentration (ABM)", linewidth=2)
# Overlay deterministic PK on same axes
ax2.plot(time, conc_det, "k--", linewidth=2, label="Drug Concentration (Deterministic)")

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
ax2.grid(True, alpha=alpha)

plt.tight_layout()
plt.show()

# Print diagnostic information
print(f"\nModel Parameters:")
print(f"Initial cells: {initial_cells}")
print(f"Birth rate: {birth_rate}, Death rate: {death_rate}")
print(f"PK decay rate (alpha): {alpha}")  # fixed to use config alpha
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
