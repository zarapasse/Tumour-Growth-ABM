from mesa import Agent, Model
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


@dataclass(frozen=True)  # ensure they cannot be modified mid simulation
class HillParams:
    """Hill kill-rate parameters."""

    K_kill: float  # max kill rate (>= 0)
    C: float  # EC50 (> 0)
    n: float  # Hill coefficient (> 0)


class TumourCell(Agent):
    """Single tumour cell agent.

    Behaviour:
    - If a birth event occurs the cell divides.
    - If a death event occurs the cell is removed.
    """

    def __init__(self, model):
        super().__init__(model)

    def step(self):
        b = self.model.birth_rate
        d = self.model.effective_death_rate
        r = b + d

        if r <= 0:
            return

        # probability of at least one event (birth or death)
        p_event = 1 - np.exp(-r * self.model.dt)

        if self.model.rng.random() < p_event:
            if self.model.rng.random() < (b / r):
                TumourCell(self.model)
            else:
                self.remove()


class TumourModel(Model):
    """
    Agent-based birth–death tumour model with optional PK/PD drug effect.

    PK:
    - Bolus doses at specified times, first-order decay with rate alpha:
      Conc(t) = sum_{doses i} amount_i * exp(-alpha * (t - t_i)) for t >= t_i

    PD:
    - Drug increases death rate only: effective_death_rate = death_rate + hill(Conc(t))
    """

    def __init__(
        self,
        initial_cells,
        birth_rate,
        death_rate,
        dt,
        alpha,
        drug_schedule=None,
        hill_params=None,
        seed=None,
    ):
        super().__init__(seed=seed)

        self.birth_rate = birth_rate
        self.death_rate = death_rate
        self.dt = dt

        self.t = 0.0

        self.alpha = alpha
        self.drug_schedule = drug_schedule if drug_schedule is not None else []
        self.hill = hill_params

        # initialise initial values (will change each step)
        self.drug_conc = 0.0
        self.effective_death_rate = death_rate

        # create initial population
        for _ in range(initial_cells):
            TumourCell(self)

    def pk_conc(self, t):
        """Compute total drug concentration at time t using exact PK decay."""
        if not self.drug_schedule:
            return 0.0
        if self.alpha == 0:
            return sum(
                amount for amount, dose_time in self.drug_schedule if t >= dose_time
            )

        total = 0.0
        for amount, dose_time in self.drug_schedule:
            if t >= dose_time:
                total += amount * np.exp(-self.alpha * (t - dose_time))
        return total

    def hill_equation(self, drug_conc):
        """Calculate drug-induced cell death using the Hill Kill function"""
        if self.hill is None or drug_conc <= 0:
            return 0.0

        x_n = drug_conc**self.hill.n
        denominator = x_n + (self.hill.C**self.hill.n)

        return float(self.hill.K_kill * (x_n / denominator))

    def step(self):
        """Advance one timestep: update PK/PD, then step agents, then advance time."""

        # 1. update drug concentration at current time
        self.drug_conc = self.pk_conc(self.t)

        # 2. update per-step birth/death probabilities based on current drug concentration
        kill = self.hill_equation(self.drug_conc)
        self.effective_death_rate = self.death_rate + kill

        # 3. step all agents in random order
        self.agents.shuffle_do("step")

        # 4. advance time
        self.t += self.dt
