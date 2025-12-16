""" This is an early script which prints a table   
Total Dose |    x_bar |   Expression
------------------------------------
Where 'Expression' is the value of the the fragility expression for given x_bar and sigma.

Doesn't use ABM, and doesn't state parameter values explicitly, so not particularly useful any more.
"""

import numpy as np
import json
from pathlib import Path



CONFIG_PATH = Path(__file__).parent / "config.json"

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)
    
# Parameters 
n = config['hill_parameters']['n']
C = config["hill_parameters"]["C"]


#total_dose_per_cycle = 40
n_doses_per_cycle = 2
n_cycles = 1
#sigma = total_dose_per_cycle/2      # for holiday example, set sigma to half of total dose
cycle_length = 12
alpha = 1
#mean_dose_per_cycle = total_dose_per_cycle / n_doses_per_cycle


T = cycle_length

# Define the function for your expression
def expression(x_bar, sigma):
    k = np.exp(-alpha * n * T / 2)
    
    term_plus  = ((x_bar + sigma)**n * k + C**n) / ((x_bar + sigma)**n + C**n)
    term_minus = ((x_bar - sigma)**n * k + C**n) / ((x_bar - sigma)**n + C**n)
    term_bar   = ((x_bar)**n * k + C**n) / ((x_bar)**n + C**n)
    
    result = np.log(term_plus) + np.log(term_minus) - 2 * np.log(term_bar)
    return result


total_doses = np.linspace(0, 200, 20)  # adjust range & number of points as needed
print(f"{'Total Dose':>12} | {'x_bar':>8} | {'Expression':>12}")
print("-"*36)



for total_dose in total_doses:
    sigma = 0.5 * total_dose
    x_bar = total_dose / n_doses_per_cycle
    val = expression(x_bar, sigma)
    print(f"{total_dose:12.2f} | {x_bar:8.2f} | {val:12.6f}")