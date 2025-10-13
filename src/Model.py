from mesa import Agent, Model
import numpy as np


class TumourCell(Agent):
    """A single tumour cell agent which may divide or die each step according to their sampled probabilities."""

    def __init__(self, model, p_birth, p_death):
        super().__init__(model)
        self.p_birth = p_birth
        self.p_death = p_death

        self.will_divide = False
        self.will_die = False

    def step(self):
        """Sample birth and death events for each cell independently to determine cell behaviour."""
        self.will_divide = np.random.rand() < self.p_birth
        self.will_die = np.random.rand() < self.p_death


class TumourModel(Model):
    """
    Container model for tumour cells using birth-death dynamics.
    Note:
        The model uses a Bernoulli approximation for events over a timestep dt:
        p = 1 - exp(-rate * dt)
    """

    def __init__(self, initial_cells, birth_rate, death_rate, dt):
        super().__init__(seed=None)
        # convert continuous rates to per-step probabilities
        self.p_birth = 1 - np.exp(-birth_rate * dt)
        self.p_death = 1 - np.exp(-death_rate * dt)

        # create initial population
        for i in range(initial_cells):
            TumourCell(self, self.p_birth, self.p_death)

    def step(self):
        """Advance the model by one timestep."""
        self.agents.shuffle_do("step")
        for agent in self.agents[
            :
        ]:  # Makes a temporary copy to allow removal while iterating
            if agent.will_divide:
                TumourCell(self, self.p_birth, self.p_death)
            if agent.will_die:
                self.agents.remove(agent)
