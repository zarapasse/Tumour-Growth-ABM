from mesa import Agent, Model
import numpy as np
import matplotlib.pyplot as plt


"""
Agent-based birth–death tumour model with optional drug effect (no PK).

A compact stochastic ABM where each tumour cell samples independent birth and
death events each timestep. The model supports an external, discrete drug
schedule that instantaneously sets a drug concentration for specified
intervals. Drug effect on proliferation is modelled with a Hill-type
inhibition function; no pharmacokinetics (PK) or spatial structure is
implemented — drug is simply "on" during scheduled intervals.

"""
class TumourCell(Agent):
    """Single tumour cell agent.

    Behaviour:
    - On each step the cell samples independent Bernoulli events based on
      the model's current p_birth and p_death.
    - If a birth event occurs the cell divides.
    - If a death event occurs the cell is removed..
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
    Population container for TumourCell agents with optional drug treatment.
    
    Responsibilities:
    - Convert continuous birth_rate/death_rate to per-step probabilities using
      p = 1 - exp(-rate * dt).
    - Maintain an optional drug_schedule (list of (start_time, duration, conc))
      and expose a simple time-varying drug_conc (no PK).
    - Compute drug inhibition using a Hill function
      and reduce the effective birth rate multiplicatively by DIP.
    - Advance time and run agent steps in randomised order.


    Attributes:
        birth_rate (float): continuous-time birth rate (per unit time).
        death_rate (float): continuous-time death rate (per unit time).
        dt (float): timestep size used to convert continuous rates to per-step probabilities.
        p_birth (float): current per-step birth probability (updated when drug is present).
        p_death (float): per-step death probability (derived from death_rate).
        drug_schedule (list): list of (start_time, duration, concentration) tuples.
        drug_conc (float): current drug concentration.
        time (float): model time (advances by dt on each step).
        E0, E1, C, n: parameters for the Hill inhibition function.
    """

    def __init__(self, initial_cells, birth_rate, death_rate, dt, drug_schedule = None):
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
        

        self.E0 = 1.0    # Baseline effect (no drug) - no inhibition
        self.E1 = 0.2    # Maximum effect (saturating drug) - 80% inhibition
        self.C = 1.0     # EC50 - concentration for half-maximal effect
        self.n = 2.0     # Hill coefficient
        

    
    def hill_equation(self, drug_conc=0.0):
        """Calculate proliferation multiplier using Hill equation: H(x) = E₀ + (xⁿ(E₁-E₀))/(xⁿ + Cⁿ)"""
        if drug_conc <= 0:
            return self.E0
        
        DIP = self.E0 + (drug_conc ** self.n * (self.E1 - self.E0)) / (drug_conc ** self.n + self.C ** self.n)
        
        return DIP
        
    def update_drug_concentration(self):
        """Advance model time by dt and set drug_conc according to the schedule.

        The method:
        - Increments self.time by self.dt.
        - Sets self.drug_conc to the concentration of the first schedule entry
          whose start <= time < start + duration, or 0.0 if none match.

        Notes:
        - This implements a simple on/off schedule (no PK or accumulation).
        """
        self.time += self.dt
        
        self.drug_conc = 0.0
        
        for start, duration, conc in self.drug_schedule:
            if start <= self.time < start + duration:
                self.drug_conc = conc
                break
        

    def step(self):
        """Advance the model by one timestep.

        Workflow:
        1. Update the drug concentration and model time.
        2. If drug is present, compute DIP via hill_equation and reduce the
           effective birth rate: effective_birth_rate = birth_rate * DIP.
           Update self.p_birth accordingly.
        3. Otherwise restore p_birth from the baseline birth_rate.
        4. Execute one step for every agent in randomized order via
           self.agents.shuffle_do("step").
           
        """
        
        self.update_drug_concentration()
        current_drug_conc = self.drug_conc
        
        if current_drug_conc > 0:
            DIP_multiplier = self.hill_equation(current_drug_conc)

            effective_birth_rate = self.birth_rate * DIP_multiplier
            self.p_birth = 1 - np.exp(-effective_birth_rate * self.dt)
        
        else:
            self.p_birth = 1 - np.exp(-self.birth_rate * self.dt)
        
        self.agents.shuffle_do("step")






# -------------- Simulation and Plotting Code --------------
# Define Parameters
initial_cells = 100
birth_rate = 0.2
death_rate = 0.1
dt = 0.1
steps = 200
T = steps * dt
n_runs = 20

# Drug schedule: list of (start_time, duration, concentration)
drug_schedule = [
    (5.0, 10.0, 1.0),   # Drug from t=5 to t-15 at concentration 1.0
    (25.0, 10.0, 2.0),  # Drug from t=25 to t=35 at concentration 2.0
]

# Run simulations with and without drug for comparison
all_populations_no_drug = np.zeros((n_runs, steps))
all_populations_with_drug = np.zeros((n_runs, steps))

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
    model = TumourModel(initial_cells, birth_rate, death_rate, dt, drug_schedule)
    population_sizes = []
    for step in range(steps):
        model.step()
        population_sizes.append(len(model.agents))
    all_populations_with_drug[run, :] = population_sizes

# Compute statistics
mean_no_drug = all_populations_no_drug.mean(axis=0)
std_no_drug = all_populations_no_drug.std(axis=0)
mean_with_drug = all_populations_with_drug.mean(axis=0)
std_with_drug = all_populations_with_drug.std(axis=0)

# Time array for plotting
time = np.arange(steps) * dt

# ========== ANALYTIC SOLUTIONS ==========
def analytic_solution_no_drug(t, initial_cells, birth_rate, death_rate):
    """Simple exponential growth without drug"""
    return initial_cells * np.exp((birth_rate - death_rate) * t)

def analytic_solution_with_drug(t, initial_cells, birth_rate, death_rate, drug_schedule, E0, E1, C, n):
    """Piecewise analytic solution with drug effects using the new Hill form"""
    population = np.zeros_like(t)
    population[0] = initial_cells
    
    for i in range(1, len(t)):
        dt_step = t[i] - t[i-1]
        
        # Determine current drug concentration
        current_drug_conc = 0.0
        for start, duration, conc in drug_schedule:
            if start <= t[i] < start + duration:
                current_drug_conc = conc
                break
        
        # Calculate effective birth rate with Hill equation
        if current_drug_conc > 0:
            # Use the same Hill equation as the ABM
            DIP_multiplier = E0 + (current_drug_conc ** n * (E1 - E0)) / (current_drug_conc ** n + C ** n)
            effective_birth_rate = birth_rate * DIP_multiplier
        else:
            effective_birth_rate = birth_rate
        
        # Exponential growth with current rates
        growth_rate = effective_birth_rate - death_rate
        population[i] = population[i-1] * np.exp(growth_rate * dt_step)
    
    return population

# Calculate analytic solutions
analytic_no_drug = analytic_solution_no_drug(time, initial_cells, birth_rate, death_rate)
analytic_with_drug = analytic_solution_with_drug(time, initial_cells, birth_rate, death_rate, 
                                                drug_schedule, E0=1.0, E1=0.2, C=1.0, n=2.0)

# ========== PLOTTING ==========
plt.figure(figsize=(12, 8))

# Plot ABM results
plt.plot(time, mean_no_drug, 'b-', label='No Drug - ABM Mean', linewidth=2)
plt.fill_between(time, mean_no_drug - std_no_drug, mean_no_drug + std_no_drug, 
                 color='blue', alpha=0.2, label='No Drug - ±1 Std Dev')
plt.plot(time, mean_with_drug, 'r-', label='With Drug - ABM Mean', linewidth=2)
plt.fill_between(time, mean_with_drug - std_with_drug, mean_with_drug + std_with_drug, 
                 color='red', alpha=0.2, label='With Drug - ±1 Std Dev')

# Plot analytic solutions
plt.plot(time, analytic_no_drug, 'b--', label='Analytic - No Drug', linewidth=2, alpha=0.7)
plt.plot(time, analytic_with_drug, 'r--', label='Analytic - With Drug', linewidth=2, alpha=0.7)

# Mark drug treatment periods
for start_time, duration, concentration in drug_schedule:
    plt.axvspan(start_time, start_time + duration, alpha=0.2, color='gray', 
                label=f'Drug Treatment (C={concentration})' if start_time == drug_schedule[0][0] else "")

plt.xlabel('Time')
plt.ylabel('Number of cells')
plt.title('Tumour Cell Population: ABM vs Analytic Solutions (Linear Scale)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Print diagnostic information
print(f"\nGrowth rate without drug: {birth_rate - death_rate:.3f}")
print(f"Expected doubling time without drug: {np.log(2)/(birth_rate - death_rate):.2f} time units")

# Check final populations
final_abm_no_drug = mean_no_drug[-1]
final_analytic_no_drug = analytic_no_drug[-1]
final_abm_with_drug = mean_with_drug[-1]
final_analytic_with_drug = analytic_with_drug[-1]

print(f"\nFinal populations:")
print(f"No Drug - ABM: {final_abm_no_drug:.0f}, Analytic: {final_analytic_no_drug:.0f}, Ratio: {final_abm_no_drug/final_analytic_no_drug:.3f}")
print(f"With Drug - ABM: {final_abm_with_drug:.0f}, Analytic: {final_analytic_with_drug:.0f}, Ratio: {final_abm_with_drug/final_analytic_with_drug:.3f}")