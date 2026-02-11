from dataclasses import dataclass

import numpy as np
from mesa import Agent, Model


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
        feasible_birth = (
            self.energy > self.model.res_params.division_threshold
        )

        if feasible_birth and (self.model.rng.random() < self.model.p_birth):
            self.energy *= 0.5
            self.model._birth_buffer.append(self.energy)


class TumourModel(Model):

    def __init__(
        self,
        initial_cells,
        birth_rate,
        death_rate,
        dt,
        initial_resources,
        initial_cell_energy,
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

        # PK/PD
        self.alpha = float(alpha)
        self.drug_schedule = list(drug_schedule) if drug_schedule else []
        self.hill = hill_params
        self.drug_conc = 0.0

        # Birth buffer (new agents added after stepping)
        self._birth_buffer = []

        for _ in range(int(initial_cells)):
            TumourCell(self, energy=float(initial_cell_energy))

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

        # Convert rates to per-step probabilities (Model-1 style)
        self.p_birth = 1.0 - np.exp(-self.birth_rate * self.dt)
        self.p_death = 1.0 - np.exp(-self.effective_death_rate * self.dt)

        # 4) Step all agents (random order)
        self._birth_buffer = []
        self.agents.shuffle_do("step")

        # 5) Add newborns after all updates
        for e in self._birth_buffer:
            TumourCell(self, energy=float(e))

        # 6) Advance time
        self.t += self.dt
