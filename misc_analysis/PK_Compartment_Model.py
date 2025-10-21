# This script plots the 1-Compartment PK model from Pierek with (α>0) and without (α=0) drug decay.
# This code generates a grid for comparison of different α values.

import numpy as np
import matplotlib.pyplot as plt

# Parameters
T = 2.5
alpha_values = [0, 0.3, 0.5]  #! Alpha values in Pierek
dose_values = np.linspace(0, 50, 200)
sigma_values = [0, 1, 3, 5, 7]  #! 1,3,5,7 in Pierek
dt = 0.01
V0 = 0.001  #! From paper (read off of graph)
n = 10  #! From paper
C = 20  #! From paper
E0 = 1  #! From paper
E1 = -1  #! From paper


# Hill function
def hill_effect(x, n, C):
    return E0 + (x**n * (E1 - E0)) / (x**n + C**n)


# Tumor growth function
def simulate_tumor(x_bar, sigma, alpha):
    t = np.arange(0, T, dt)
    x_t = np.zeros_like(t)  # Makes t number of zeros
    for i, ti in enumerate(t):
        if 0 <= ti < T / 2:
            x_t[i] = (x_bar + sigma) * np.exp(-alpha * ti)
        else:
            x_t[i] = (x_bar - sigma) * np.exp(-alpha * (ti - T / 2))

    V = np.zeros_like(t)
    V[0] = V0
    for i in range(1, len(t)):
        V[i] = V[i - 1] * np.exp(hill_effect(x_t[i - 1], n, C) * dt)

    return t, x_t, V


# Fragility Function
def fragility_function(x_bar, sigma, alpha):
    _, _, V_even = simulate_tumor(x_bar, 0, alpha)
    _, _, V_uneven = simulate_tumor(x_bar, sigma, alpha)
    fragility = (V_uneven[-1] - V_even[-1]) / V0
    return fragility


# Function to compute Hill curve for given alpha
def compute_hill_curve(alpha):
    avg_concentrations = []  # Takes average concentration of drug for EVEN dosing
    for x_bar in dose_values:
        t = np.arange(0, T, dt)
        x_t = np.zeros_like(t)
        for i, ti in enumerate(t):
            if 0 <= ti < T / 2:
                x_t[i] = x_bar * np.exp(-alpha * ti)
            else:
                x_t[i] = x_bar * np.exp(-alpha * (ti - T / 2))
        avg_x = np.mean(x_t)
        avg_concentrations.append(avg_x)
    return hill_effect(np.array(avg_concentrations), n, C)


# Plot for different alpha

fig, axs = plt.subplots(3, 4, figsize=(20, 12))

for row, alpha in enumerate(alpha_values):

    # Compute tumor trajectory and drug concentrations
    t_even, x_even, v_even = simulate_tumor(x_bar=25, sigma=0, alpha=alpha)
    t_uneven, x_uneven, v_uneven = simulate_tumor(x_bar=25, sigma=15, alpha=alpha)

    # Compute Hill curve
    hill_curve = compute_hill_curve(alpha)

    # Compute fragility analysis
    fragility_results = {}
    for sigma in sigma_values:
        fragility_list = []
        for x_bar in dose_values:
            _, _, V_even = simulate_tumor(x_bar, 0, alpha)
            _, _, V_uneven = simulate_tumor(x_bar, sigma, alpha)
            frag = (V_uneven[-1] - V_even[-1]) / V0
            fragility_list.append(frag)
        fragility_results[sigma] = np.array(fragility_list)

    axs[row, 0].plot(t_even, v_even, lw=2, label="Even dosing")
    axs[row, 0].plot(t_uneven, v_uneven, lw=2, label="Uneven dosing")
    axs[row, 0].set_xlabel("Time")
    axs[row, 0].set_ylabel("Tumor volume V(t)")
    axs[row, 0].set_title(f"Tumor Trajectory (α={alpha})")
    axs[row, 0].legend()
    axs[row, 0].grid(True)

    axs[row, 1].plot(t_even, x_even, lw=2, label="Even dosing")
    axs[row, 1].plot(t_uneven, x_uneven, lw=2, label="Uneven dosing")
    axs[row, 1].axvline(T / 2, color="gray", linestyle="--", alpha=0.5)
    axs[row, 1].set_xlabel("Time")
    axs[row, 1].set_ylabel("Drug concentration")
    axs[row, 1].set_title(f"Drug Concentration (α={alpha})")
    axs[row, 1].legend()
    axs[row, 1].grid(True)

    axs[row, 2].plot(dose_values, hill_curve, lw=2, color="green")
    axs[row, 2].axhline(0, color="gray", linestyle="--", alpha=0.5)
    axs[row, 2].set_xlabel("Average blood dose concentration for EVEN dosing")
    axs[row, 2].set_ylabel("Hill function")
    axs[row, 2].set_title(f"Hill Function (α={alpha})")
    axs[row, 2].grid(True)

    for sigma in sigma_values:
        axs[row, 3].plot(
            dose_values, fragility_results[sigma], lw=2, label=f"σ={sigma}"
        )
    axs[row, 3].axhline(0, color="gray", linestyle="--")
    axs[row, 3].set_xlabel("Drug dose x̄")
    axs[row, 3].set_ylabel("Fragility")
    axs[row, 3].set_title(f"Fragility vs Dose (α={alpha})")
    axs[row, 3].legend()
    axs[row, 3].grid(True)

plt.tight_layout()

plt.savefig("PK_Model_Results.png", dpi=300, bbox_inches="tight")
print("Figure saved as 'PK_Model_Results.png'")

plt.show()
