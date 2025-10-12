from mesa import Agent, Model
import numpy as np

class TumourCell(Agent):
    def __init__(self, model, p_birth, p_death):
        super().__init__(model)
        self.p_birth = p_birth
        self.p_death = p_death
        
    def step(self):
        self.will_divide = True if np.random.rand() < self.p_birth else False
        self.will_die = True if np.random.rand() < self.p_death else False
            
class TumourModel(Model):
    def __init__(self, initial_cells, birth_rate, death_rate, dt):
        super().__init__(seed = None)
        
        self.p_birth = 1 - np.exp(-birth_rate * dt)
        self.p_death = 1 - np.exp(-death_rate * dt)
        
        for i in range(initial_cells):
            TumourCell(self, self.p_birth, self.p_death)
            
    def step(self):
        self.agents.shuffle_do("step")
        for agent in self.agents[:]:
            if agent.will_divide:
                TumourCell(self, self.p_birth, self.p_death)
            if agent.will_die:
                self.agents.remove(agent)
        
        