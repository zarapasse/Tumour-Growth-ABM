from mesa import Agent, Model
from mesa.datacollection import DataCollector
import numpy as np

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
        self.cell_type = "sensitive"
        self.p_birth = model.p_birth
        self.p_death = model.p_death

    def step(self):
        if np.random.rand() < self.model.p_birth:
            if (
                self.model.enable_resistance
                and np.random.rand() < self.model.p_mutation
            ):
                ResistantTumourCell(self.model)

            else:
                TumourCell(self.model)

        if np.random.rand() < self.model.p_death:
            self.remove()


class ResistantTumourCell(TumourCell):
    """A drug-resistant tumour cell that is unaffected by the drug."""

    def __init__(self, model):
        super().__init__(model)
        self.cell_type = "resistant"

        self.p_birth = model.p_birth
        self.p_death = 1 - np.exp(
            -model.death_rate * model.dt
        )  # baseline death only (no drug effect)

    def step(self):
        if np.random.rand() < self.p_birth:
            ResistantTumourCell(self.model)
        if np.random.rand() < self.p_death:
            self.remove()


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
        alpha,
        p_mutation,
        drug_schedule=None,
        hill_params=None,
        enable_resistance=False,
    ):
        super().__init__(seed=None)
        self.enable_resistance = enable_resistance

        self.birth_rate = birth_rate
        self.death_rate = death_rate
        self.dt = dt
        self.p_birth = 1 - np.exp(-birth_rate * dt)
        self.p_death = 1 - np.exp(-death_rate * dt)
        self.p_mutation = p_mutation if enable_resistance else 0.0

        # create initial population
        for _ in range(initial_cells):
            TumourCell(self)

        # Drug parameters
        self.drug_schedule = drug_schedule if drug_schedule is not None else []
        self.drug_conc = 0.0
        self.time = 0.0
        self.alpha = alpha

        self.E0, self.E1, self.C, self.n = (
            hill_params["E0"],
            hill_params["E1"],
            hill_params["C"],
            hill_params["n"],
        )

        # Mesa DataCollector to track populations
        self.datacollector = DataCollector(
            model_reporters={
                "Time": lambda m: m.time,
                "Sensitive": lambda m: sum(
                    1 for a in m.agents if type(a) is TumourCell
                ),
                "Resistant": lambda m: sum(
                    1 for a in m.agents if isinstance(a, ResistantTumourCell)
                ),
                "Total": lambda m: len(m.agents),
                "ResistantFraction": lambda m: (
                    sum(1 for a in m.agents if isinstance(a, ResistantTumourCell))
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
        """Advance the model by one timestep.

        Workflow:
        1. Update the drug concentration and model time.
        2. If drug is present, drug induced death via hill_equation and increase
           the effective death rate: effective_death_rate = death_rate + kill_rate.
           Update self.p_death accordingly.
        3. Otherwise restore p_death from the baseline death_rate.
        4. Always keep p_birth at baseline (drug affects death only).
        """

        current_drug_conc = self.drug_conc
        self.update_drug_concentration()

        # baseline p_birth
        self.p_birth = 1 - np.exp(-self.birth_rate * self.dt)

        if current_drug_conc > 0:
            kill_rate = self.hill_equation(current_drug_conc)
            effective_death_rate = self.death_rate + kill_rate
            self.p_death = 1 - np.exp(-effective_death_rate * self.dt)
        else:
            self.p_death = 1 - np.exp(-self.death_rate * self.dt)

        self.agents.shuffle_do("step")
        self.datacollector.collect(self)
