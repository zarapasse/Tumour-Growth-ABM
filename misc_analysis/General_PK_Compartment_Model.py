#No cap general PK compartment model simulation and analysis

import numpy as np
import matplotlib.pyplot as plt

# Parameters
T = 2             
alpha_values = [0.1, 0.3, 0.5]  #! Alpha values to compare in Pierek    
dose_values = np.linspace(0, 50, 200)
sigma_values = [7, 9, 11, 13, 15]  #! 1,3,5,7 in Pierek
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
def simulate_tumor(x_bar, sigma, alpha):
    t = np.arange(0, T, dt)
    x_t = np.zeros_like(t)  #Makes t number of zeros
    for i, ti in enumerate(t):
        # First dose contribution (always present, decaying from t=0)
        dose1_contrib = (x_bar + sigma) * np.exp(-alpha * ti)
        
        # Second dose contribution (only present after t >= T/2, decaying from T/2)
        if ti >= T/2:
            dose2_contrib = (x_bar - sigma) * np.exp(-alpha * (ti - T/2))
            x_t[i] = dose1_contrib + dose2_contrib
        else:
            x_t[i] = dose1_contrib
    
    V = np.zeros_like(t)
    V[0] = V0
    for i in range(1, len(t)):
        V[i] = V[i-1] * np.exp(hill_effect(x_t[i-1], n, C) * dt)
    
    return t, x_t, V

# Fragility Function
def fragility_function(x_bar, sigma, alpha):
    _, _, V_even = simulate_tumor(x_bar, 0, alpha)
    _, _, V_uneven = simulate_tumor(x_bar, sigma, alpha)
    fragility = (V_uneven[-1] - V_even[-1]) / V0
    return fragility

# Function to compute Hill curve for given alpha AT t=0.1 
def compute_hill_curve(alpha):
    x_vals = []
    for x_bar in dose_values:
        x = x_bar * np.exp(-alpha*0.1)  # concentration at t = 0.1
        x_vals.append(x)
    return np.array(dose_values), hill_effect(np.array(x_vals), n, C)


# Plot for different alpha
import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(20, 12))
gs_main = gridspec.GridSpec(3, 4, figure=fig, hspace=0.4, wspace=0.3)

for row, alpha in enumerate(alpha_values):

    # Compute tumor trajectory and drug concentrations
    t_even, x_even, v_even = simulate_tumor(x_bar=25, sigma=0, alpha=alpha)
    t_uneven, x_uneven, v_uneven = simulate_tumor(x_bar=25, sigma=25, alpha=alpha)

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

    # Column 0: Tumor trajectory
    ax0 = fig.add_subplot(gs_main[row, 0])
    ax0.plot(t_even, v_even, lw=2, label="Even dosing")
    ax0.plot(t_uneven, v_uneven, lw=2, label="Uneven dosing")
    ax0.set_xlabel("Time")
    ax0.set_ylabel("Tumor volume V(t)")
    ax0.set_title(f"Tumor Trajectory (α={alpha})")
    ax0.legend()
    ax0.grid(True)

    # Column 1: Drug concentration - create nested gridspec for stacked plots
    gs_drug = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_main[row, 1], hspace=0.05)
    
    # Even dosing (top)
    ax1a = fig.add_subplot(gs_drug[0])
    ax1a.fill_between(t_even, x_even, alpha=0.3, color='C0')
    ax1a.plot(t_even, x_even, lw=2, color='C0', label='Even')
    ax1a.axvline(T / 2, color="gray", linestyle="--", alpha=0.5)
    ax1a.set_ylabel("Conc.")
    ax1a.set_title(f"Drug Concentration (α={alpha})")
    ax1a.legend(loc='upper right')
    ax1a.grid(True)
    ax1a.tick_params(labelbottom=False)  # Remove x-axis labels for top plot
    
    # Uneven dosing (bottom)
    ax1b = fig.add_subplot(gs_drug[1], sharex=ax1a)
    ax1b.fill_between(t_uneven, x_uneven, alpha=0.3, color='C1')
    ax1b.plot(t_uneven, x_uneven, lw=2, color='C1', label='Uneven')
    ax1b.axvline(T / 2, color="gray", linestyle="--", alpha=0.5)
    ax1b.set_xlabel("Time")
    ax1b.set_ylabel("Conc.")
    ax1b.legend(loc='upper right')
    ax1b.grid(True)

    # Column 2: Hill curve
    ax2 = fig.add_subplot(gs_main[row, 2])
    x_vals, hill_vals = compute_hill_curve(alpha)
    ax2.plot(x_vals, hill_vals, lw=2, color="green")
    ax2.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Drug dose x̄")
    ax2.set_ylabel("H(x̄)")
    ax2.set_title(f"DIP Curve (α={alpha}) at t=0.1")
    ax2.grid(True)

    # Column 3: Fragility
    ax3 = fig.add_subplot(gs_main[row, 3])
    for sigma in sigma_values:
        ax3.plot(dose_values, fragility_results[sigma], lw=2, label=f"σ={sigma}")
    ax3.axhline(0, color="gray", linestyle="--")
    ax3.set_xlabel("Drug dose x̄")
    ax3.set_ylabel("Fragility")
    ax3.set_title(f"Fragility vs Dose (α={alpha})")
    ax3.legend()
    ax3.grid(True)

plt.savefig("misc_analysis/graphs/General_PK_Results.png", dpi=300, bbox_inches="tight")
print("Figure saved as 'General_PK_Results.png'")

plt.show()

