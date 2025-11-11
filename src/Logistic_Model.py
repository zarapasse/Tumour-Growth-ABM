from mesa import Agent, Model
import numpy as np
import matplotlib.pyplot as plt


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

    def __init__(self, initial_cells, birth_rate, death_rate, dt, resources, resource_influx, single_feed):
        super().__init__(seed=None)
        
        self.resources = resources
        self.resource_influx = resource_influx
        self.single_feed = single_feed
        
        # convert continuous rates to per-step probabilities
        self.p_birth = 1 - np.exp(-birth_rate * dt)
        self.p_death = 1 - np.exp(-death_rate * dt)

        # create initial population
        for i in range(initial_cells):
            TumourCell(self, self.p_birth, self.p_death)

    def step(self):
        """Advance the model by one timestep.
        Each cell tries to divide if it reaches probability AND resources are available."""
        
        self.resources += self.resource_influx 

        self.agents.shuffle_do("step")     # Each cell samples whether it will divide or die 
        for agent in list(self.agents):      
            if agent.will_die:
                self.agents.remove(agent)
                
        maintenance_cost = len(self.agents)*self.single_feed
        
        if self.resources < maintenance_cost:
            surviving_cells = int(self.resources // self.single_feed)
            cells_to_remove = len(self.agents) - surviving_cells
            agents_list = list(self.agents)
            np.random.shuffle(agents_list)
            for i in range(cells_to_remove):
                self.agents.remove(agents_list[i])
                
        self.resources = self.resources - len(self.agents)*self.single_feed
        for agent in list(self.agents):
            if agent.will_divide and self.resources >= self.single_feed:
                TumourCell(self, self.p_birth, self.p_death)                
                self.resources -=  self.single_feed
                
                
                
                


# ------------------- Single Large-Population Run ------------------- #
timesteps = 300
initial_cells = 5000  # large population
model = TumourModel(
    initial_cells=initial_cells,
    birth_rate=0.25,
    death_rate=0.1,
    dt=0.05,
    resources=initial_cells*1.1,  # scale resources with population
    resource_influx=initial_cells*1.05,  # modest influx
    single_feed=1
)

cell_counts = []
resources = []

for t in range(timesteps):
    model.step()
    cell_counts.append(len(model.agents))
    resources.append(model.resources)

# ------------------- Plot ------------------- #
plt.figure(figsize=(10,5))
plt.plot(cell_counts, label='Cell count')
plt.plot(resources, label='Resources', linestyle='--')
plt.xlabel('Time step')
plt.ylabel('Count')
plt.title('Single Large-Population Run: Logistic-like Behaviour Emerges')
plt.legend()
plt.show()