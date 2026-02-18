from mesa import Agent, Model
from mesa.datacollection import DataCollector
import numpy as np
from dataclasses import dataclass

"""
Agent-based birth–death tumour model with optional drug effect.

Pharmacology:
- Drug is administered as bolus doses at specific times and follows
  first-order decay with rate alpha (PK).
- Drug action increases death rate: the effective death rate is death_rate + kill_rate(Conc), 
  where kill_rate(Conc) is a Hill function of the instantaneous concentration Conc(t).
- The birth rate is unaffected by the drug.
"""

@dataclass(frozen=True) # ensure they cannot be modified mid simulation
class HillParams:
    """Hill kill-rate parameters."""
    K_kill: float  # max kill rate (>= 0)
    C: float       # EC50 (> 0)
    n: float       # Hill coefficient (> 0)
class TumourCell(Agent):
    """Single tumour cell agent.

    Behaviour:
    - On each step the cell samples independent Bernoulli events based on
      the model's current p_birth and p_death.
    - If a birth event occurs the cell divides.
    - If a death event occurs the cell is removed.
    """
    
    cell_type = "sensitive"

    def __init__(self, model):
        super().__init__(model)

    def step(self):
        b=self.model.birth_rate
        d=self.model.effective_death_rate
        r = b + d
        
        if r <= 0:
            return
        
        #probability of at least one event (birth or death)
        p_event = 1 - np.exp(-r * self.model.dt)
        
        if self.model.rng.random() < p_event:  
            if self.model.rng.random() < (b/r):
                if self.model.enable_resistance and (self.model.rng.random() < self.model.p_mutation):
                    self.model.spawn_resistant()
                else:
                    self.model.spawn_sensitive()
            else:
                self.model.kill_cell(self)


class ResistantTumourCell(Agent):
    """A drug-resistant tumour cell that is unaffected by the drug."""

    cell_type = "resistant"
    
    def __init__(self, model):
        super().__init__(model)
        
    def step(self):
        b=self.model.birth_rate * (1.0 - self.model.fitness_cost)  # resistant cells have a fitness cost
        d=self.model.death_rate  # resistant cells are unaffected by drug, so use base death rate
        r = b + d
        
        if r <= 0:
            return
        
        #probability of at least one event (birth or death)
        p_event = 1 - np.exp(-r * self.model.dt)
        
        if self.model.rng.random() < p_event:  
            if self.model.rng.random() < (b/r):
                self.model.spawn_resistant()
            else:
                self.model.kill_cell(self)
class TumourModel(Model):
    """
    Agent-based birth–death tumour model with optional drug resistance and PK/PD drug effect.
    """

    def __init__(
        self,
        initial_cells,
        birth_rate,
        death_rate,
        dt,
        alpha,
        fitness_cost,
        drug_schedule=None,
        hill_params=None,
        seed=None,
        enable_resistance=False,
        p_mutation=0.0,
        initial_resistant_fraction=0.0,
    ):
        super().__init__(seed=seed)
        
        self.birth_rate = birth_rate
        self.death_rate = death_rate
        self.dt = dt
        
        self.t = 0.0
        
        self.alpha = alpha
        self.drug_schedule = drug_schedule if drug_schedule is not None else []
        self.hill = hill_params
        self.fitness_cost = fitness_cost
        
        self.enable_resistance = enable_resistance
        self.p_mutation = p_mutation if self.enable_resistance else 0.0
        self.initial_resistant_fraction = float(initial_resistant_fraction)

        #initial values 
        self.drug_conc = 0.0
        self.effective_death_rate = self.death_rate
        
        self.n_sensitive = 0
        self.n_resistant = 0
        
        # create initial population with specified resistant fraction
        n_resistant = int(initial_cells * self.initial_resistant_fraction)
        n_sensitive = initial_cells - n_resistant
        
        for _ in range(n_sensitive):
            self.spawn_sensitive()
        for _ in range(n_resistant):
            self.spawn_resistant()            

        self.datacollector = DataCollector(
            model_reporters={
                "t": lambda m: m.t,
                "Sensitive": lambda m: m.n_sensitive,
                "Resistant": lambda m: m.n_resistant,
                "Total": lambda m: m.n_sensitive + m.n_resistant,
                "ResistantFraction": lambda m: (
                    m.n_resistant / max(1, m.n_sensitive + m.n_resistant)
                ),
                "DrugConc": lambda m: m.drug_conc,
                "KillRate": lambda m: m.effective_death_rate - m.death_rate,
            }
        )
        self.datacollector.collect(self)   # collect t=0
        
    def spawn_sensitive(self):
        """Spawn a new sensitive cell."""
        TumourCell(self)
        self.n_sensitive += 1
        
    def spawn_resistant(self):
        """Spawn a new resistant cell."""
        ResistantTumourCell(self)
        self.n_resistant += 1
        
    def kill_cell(self, cell):
        """Kill a cell and update counts."""
        if cell.cell_type == "sensitive":
            self.n_sensitive -= 1
        else:
            self.n_resistant -= 1
        cell.remove()

    def pk_conc(self, t):
        """Compute total drug concentration at time t using exact PK decay."""
        if not self.drug_schedule:
            return 0.0
        if self.alpha == 0:
            return sum(amount for amount, dose_time in self.drug_schedule if t >= dose_time)

        total = 0.0
        for amount, dose_time in self.drug_schedule:
            if t >= dose_time:
                total += amount * np.exp(-self.alpha * (t - dose_time))
        return total
    
    def hill_equation(self, drug_conc):
        """Calculate drug-induced cell death using the Hill Kill function"""
        if self.hill is None or drug_conc <= 0:
            return 0.0
        
        x_n = drug_conc ** self.hill.n
        denominator = x_n + (self.hill.C ** self.hill.n)
        
        return float(self.hill.K_kill * (x_n / denominator))
    
    def step(self):
        """Advance one timestep: update PK/PD, then step agents, then advance time."""

        #1. update drug concentration at current time
        self.drug_conc = self.pk_conc(self.t)
        
        #2. update per-step birth/death probabilities based on current drug concentration
        kill = self.hill_equation(self.drug_conc)
        self.effective_death_rate = self.death_rate + kill
        
        #3. step all agents in random order
        self.agents.shuffle_do("step")
        
        #4. advance time
        self.t += self.dt
    
        self.datacollector.collect(self)
    