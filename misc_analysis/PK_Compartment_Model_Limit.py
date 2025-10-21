# This script plots the 1-Compartment PK model from Pierek with (α>0) and without (α=0) drug decay.
# This code generates a plot for only one α value.

import numpy as np
import matplotlib.pyplot as plt

# Parameters
T = 2.5             
alpha = 5   #! 0,0.3,0.5 in Pierek    
dose_values = np.linspace(0, 50, 200)
sigma_values = [0,1,3,5,7] #! 1,3,5,7 in Pierek
dt = 0.01
V0 = 0.001  #! From paper (read off of graph)
n = 10 #! From paper
C = 20 #! From paper
E0 = 1  #! From paper
E1 = -1  #! From paper

# Hill function
def hill_effect(x, n, C):
    return E0 + (x**n * (E1 - E0)) / (x**n + C**n)

# Tumor growth function
def simulate_tumor(x_bar, sigma):
    t = np.arange(0, T, dt)
    x_t = np.zeros_like(t)  #Makes t number of zeros
    for i, ti in enumerate(t):
        if 0 <= ti < T/2:
            x_t[i] = (x_bar + sigma) * np.exp(-alpha * ti)
        else:
            x_t[i] = (x_bar - sigma) * np.exp(-alpha * (ti - T/2))
    
    V = np.zeros_like(t)
    V[0] = V0
    for i in range(1, len(t)):
        V[i] = V[i-1] * np.exp(hill_effect(x_t[i-1], n, C) * dt)
    
    return t, x_t, V

# Fragility Function
def fragility_function(x_bar, sigma):
    _, _, V_even = simulate_tumor(x_bar, 0)
    _, _, V_uneven = simulate_tumor(x_bar, sigma)
    fragility = (V_uneven[-1] - V_even[-1]) / V0
    return fragility



#Finding values for plots

# Tumor trajectory and drug concentrations to mimic graphs in paper
t_even, x_even, v_even = simulate_tumor(x_bar=25, sigma=0)
t_uneven, x_uneven, v_uneven = simulate_tumor(x_bar=25, sigma=15)

# Hill function 
# Compute average drug concentration for each dose using current alpha
avg_concentrations = []
for x_bar in dose_values:
    t = np.arange(0, T, dt)
    x_t = np.zeros_like(t)
    for i, ti in enumerate(t):
        if 0 <= ti < T/2:
            x_t[i] = x_bar * np.exp(-alpha * ti)
        else:
            x_t[i] = x_bar * np.exp(-alpha * (ti - T/2))
    # Average drug concentration over time
    avg_x = np.mean(x_t)
    avg_concentrations.append(avg_x)

hill_curve = hill_effect(np.array(avg_concentrations), n, C)

# Fragility analysis
fragility_results = {}

for sigma in sigma_values:
    fragility_list = []
    for x_bar in dose_values:
        _, _, V_even = simulate_tumor(x_bar, 0)
        _, _, V_uneven = simulate_tumor(x_bar, sigma)
        frag = (V_uneven[-1] - V_even[-1]) / V0
        fragility_list.append(frag)
    fragility_results[sigma] = np.array(fragility_list)



# Plot 
fig, axs = plt.subplots(1, 4, figsize=(20, 5)) 

# Tumor trajectory
axs[0].plot(t_even, v_even, lw=2, label='Even dosing')
axs[0].plot(t_uneven, v_uneven, lw=2, label='Uneven dosing')
axs[0].set_xlabel('Time')
axs[0].set_ylabel('Tumor volume V(t)')
axs[0].set_title('Tumor Trajectory for x̄=25')
axs[0].legend()
axs[0].grid(True)

# Drug concentration
axs[1].plot(t_even, x_even, lw=2, label='Even dosing')
axs[1].plot(t_uneven, x_uneven, lw=2, label='Uneven dosing')
axs[1].axvline(T/2, color='gray', linestyle='--', alpha=0.5)
axs[1].set_xlabel('Time')
axs[1].set_ylabel('Drug concentration')
axs[1].set_title('Drug Concentration Over Time')
axs[1].legend()
axs[1].grid(True)

# Hill function vs Dose 
axs[2].plot(dose_values, hill_curve, lw=2, color='green', label=f'α={alpha}')
axs[2].axhline(0, color='gray', linestyle='--', alpha=0.5)
axs[2].set_xlabel('Drug dose x̄')
axs[2].set_ylabel('Hill Function')
axs[2].set_title(f'Hill Function (α={alpha})')
axs[2].legend()
axs[2].grid(True)


# Fragility curve
for sigma in sigma_values:
    axs[3].plot(dose_values, fragility_results[sigma],
                lw=2, label=f'σ={sigma}')
axs[3].axhline(0, color='gray', linestyle='--')
axs[3].set_xlabel('Average dose x̄')
axs[3].set_ylabel('Fragility')
axs[3].set_title('Fragility vs Dose')
axs[3].legend()
axs[3].grid(True)

plt.tight_layout()
plt.show()

