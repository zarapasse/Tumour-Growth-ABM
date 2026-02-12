from dataclasses import dataclass

import numpy as np
from mesa import Agent, Model
from mesa.datacollection import DataCollector


@dataclass(frozen=True)
class HillParams:
    """Hill kill-rate parameters."""

    K_kill: float
    C: float
    n: float


@dataclass(frozen=True)
class ResourceParams:
    """Resource/energy rule parameters."""

    energy_capacity: float
    division_threshold: float
    maintenance_cost: float
    resource_influx: float


class TumourCell(Agent):
    """Single tumour cell with internal energy."""

    cell_type = "sensitive"

    def __init__(self, model, energy):
        super().__init__(model)
        self.energy = float(energy)

    def step(self):
        maintenance = self.model.res_params.maintenance_cost
        energy_capacity = self.model.res_params.energy_capacity

        # 1) Metabolic maintenance
        self.energy -= maintenance

        # 2) Energy-depletion death
        if self.energy <= 0.0:
            self.remove()
            return

        # 3) Stochastic death (independent Bernoulli trial)
        if self.model.rng.random() < self.model.p_death:
            self.remove()
            return

        # 4) Resource uptake (bounded by remaining capacity + availability)
        capacity = energy_capacity - self.energy
        if capacity > 0.0 and self.model.resources > 0.0:
            uptake = min(self.model.resources, capacity)
            self.energy += uptake
            self.model.resources -= uptake

        # 5) Division attempt (independent Bernoulli trial, gated by feasibility)
        feasible_birth = self.energy > self.model.res_params.division_threshold

        if feasible_birth and (self.model.rng.random() < self.model.p_birth):
            self.energy *= 0.5
            if self.model.rng.random() < self.model.p_mutation:
                self.model._birth_buffer_resistant.append(self.energy)
            else:
                self.model._birth_buffer.append(self.energy)


class TumourResistantCell(Agent):
    """A resistant tumour cell unaffected by drug."""

    cell_type = "resistant"

    def __init__(self, model, energy):
        super().__init__(model)
        self.energy = float(energy)
        self.p_baseline_death = self.model.p_baseline_death

    def step(self):
        maintenance = self.model.res_params.maintenance_cost
        energy_capacity = self.model.res_params.energy_capacity

        # 1) Metabolic maintenance
        self.energy -= maintenance

        # 2) Energy-depletion death
        if self.energy <= 0.0:
            self.remove()
            return

        # 3) Stochastic death (unaffected by drug)
        if self.model.rng.random() < self.p_baseline_death:
            self.remove()
            return

        # 4) Resource uptake (bounded by remaining capacity + availability)
        capacity = energy_capacity - self.energy
        if capacity > 0.0 and self.model.resources > 0.0:
            uptake = min(self.model.resources, capacity)
            self.energy += uptake
            self.model.resources -= uptake

        # 5) Division attempt (independent Bernoulli trial, gated by feasibility)
        feasible_birth = self.energy > self.model.res_params.division_threshold

        if feasible_birth and (self.model.rng.random() < self.model.p_birth):
            self.energy *= 0.5
            self.model._birth_buffer_resistant.append(self.energy)


class ResistantTumourModel(Model):

    def __init__(
        self,
        initial_cells,
        birth_rate,
        death_rate,
        dt,
        initial_resources,
        initial_cell_energy,
        p_mutation,
        initial_resistant_fraction,
        res_params=None,
        alpha=0.0,
        hill_params=None,
        drug_schedule=None,
        seed=None,
    ):
        super().__init__(seed=seed)

        self.dt = float(dt)
        self.t = 0.0

        self.resources = float(initial_resources)
        self.res_params = res_params

        self.birth_rate = float(birth_rate)
        self.death_rate = float(death_rate)

        # updated each step based on PK/PD
        self.p_birth = 0.0
        self.p_death = 0.0

        self.p_baseline_death = 1.0 - np.exp(-self.death_rate * self.dt)
        self.p_mutation = float(p_mutation)
        self.effective_death_rate = self.death_rate

        # PK/PD
        self.alpha = float(alpha)
        self.drug_schedule = list(drug_schedule) if drug_schedule else []
        self.hill = hill_params
        self.drug_conc = 0.0

        # Birth buffer (new agents added after stepping)
        self._birth_buffer = []
        self._birth_buffer_resistant = []

        initial_resistant_cells = int(round(initial_cells * initial_resistant_fraction))
        initial_sensitive_cells = int(initial_cells) - initial_resistant_cells

        for _ in range(initial_sensitive_cells):
            TumourCell(self, energy=float(initial_cell_energy))

        for _ in range(initial_resistant_cells):
            TumourResistantCell(self, energy=float(initial_cell_energy))

        # initialise counts
        self.n_sensitive = sum(a.cell_type == "sensitive" for a in self.agents)
        self.n_resistant = len(self.agents) - self.n_sensitive

        # initialise PK/PD at t=0 so t=0 row is meaningful
        self.drug_conc = self.pk_conc(self.t)
        kill = self.hill_equation(self.drug_conc)
        self.effective_death_rate = self.death_rate + kill
        self.p_birth = 1.0 - np.exp(-self.birth_rate * self.dt)
        self.p_death = 1.0 - np.exp(-self.effective_death_rate * self.dt)

        self.datacollector = DataCollector(
            model_reporters={
                "t": lambda m: m.t,
                "Sensitive": lambda m: m.n_sensitive,
                "Resistant": lambda m: m.n_resistant,
                "Total": lambda m: m.n_sensitive + m.n_resistant,
                "ResistantFraction": lambda m: m.n_resistant
                / max(1, (m.n_sensitive + m.n_resistant)),
                "DrugConc": lambda m: m.drug_conc,
                "KillRate": lambda m: m.effective_death_rate - m.death_rate,
            }
        )

        self.datacollector.collect(self)  # collect t=0

    def pk_conc(self, t):
        """Exact PK concentration at time t from bolus doses with exponential decay."""
        if not self.drug_schedule:
            return 0.0
        if self.alpha == 0.0:
            # No decay: step accumulation
            return float(
                sum(
                    amount for amount, dose_time in self.drug_schedule if t >= dose_time
                )
            )

        total = 0.0
        for amount, dose_time in self.drug_schedule:
            if t >= dose_time:
                total += amount * np.exp(-self.alpha * (t - dose_time))
        return float(total)

    def hill_equation(self, drug_conc):
        if self.hill is None or drug_conc <= 0.0:
            return 0.0
        x_n = drug_conc**self.hill.n
        denom = x_n + (self.hill.C**self.hill.n)
        if denom <= 0.0:
            return 0.0
        return float(self.hill.K_kill * (x_n / denom))

    def step(self):

        # 1) Resource influx at the start of the timestep
        self.resources += self.res_params.resource_influx

        # 2) PK at current time
        self.drug_conc = self.pk_conc(self.t)

        # 3) PD -> effective death rate + heuristic per-step probabilities
        kill = self.hill_equation(self.drug_conc)
        self.effective_death_rate = self.death_rate + kill

        # Convert rates to per-step probabilities
        self.p_birth = 1.0 - np.exp(-self.birth_rate * self.dt)
        self.p_death = 1.0 - np.exp(-self.effective_death_rate * self.dt)

        # 4) Step all agents (random order)
        self._birth_buffer = []
        self._birth_buffer_resistant = []
        self.agents.shuffle_do("step")

        # 5) Add newborns after all updates
        for e in self._birth_buffer:
            TumourCell(self, energy=float(e))

        for e in self._birth_buffer_resistant:
            TumourResistantCell(self, energy=float(e))

        # 6) Advance time
        self.t += self.dt

        self.n_sensitive = sum(a.cell_type == "sensitive" for a in self.agents)
        self.n_resistant = len(self.agents) - self.n_sensitive

        # 6) recompute PK/PD at the NEW time for consistent reporting
        self.drug_conc = self.pk_conc(self.t)
        kill = self.hill_equation(self.drug_conc)
        self.effective_death_rate = self.death_rate + kill

        # 7) collect the state at time t
        self.datacollector.collect(self)
