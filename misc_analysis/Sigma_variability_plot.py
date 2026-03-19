import numpy as np
import matplotlib.pyplot as plt

# ---------------- Parameters ---------------- #
T = 2.0
dt = 0.01
V0 = 0.001

# Fast PK regime
alpha = 5

# Mean dose range
dose_values = np.linspace(5, 50, 300)

# Relative variability levels: sigma = c * xbar
sigma_fractions = [0.0, 0.25, 0.5, 0.75, 1]

# PD (Hill kill)
n = 5
C = 25
K_kill = 2.5

# Tumour intrinsic growth
k = 0.25


# ---------------- Helper functions ---------------- #
def gamma_kill(x):
    """Hill kill function gamma(x)."""
    x = np.maximum(x, 0.0)
    return K_kill * (x**n) / (x**n + C**n)


def pk_profile(t, x_bar, sigma, alpha):
    """
    Fast-PK two-dose profile:
    - first half-cycle dose = x_bar + sigma
    - second half-cycle dose = x_bar - sigma
    """
    dose1 = x_bar + sigma
    dose2 = max(x_bar - sigma, 0.0)  # avoid negative doses

    x_t = np.zeros_like(t)

    for i, ti in enumerate(t):
        if ti < T / 2:
            x_t[i] = dose1 * np.exp(-alpha * ti)
        else:
            x_t[i] = dose2 * np.exp(-alpha * (ti - T / 2))

    return x_t


def simulate_tumor(x_bar, sigma, alpha):
    """Simulate tumour dynamics under PK-PD model."""
    t = np.arange(0, T + dt, dt)
    x_t = pk_profile(t, x_bar, sigma, alpha)

    V = np.zeros_like(t)
    V[0] = V0

    # Exact step for dV/dt = (k - gamma(x(t)))V
    for i in range(1, len(t)):
        g = gamma_kill(x_t[i - 1])
        V[i] = V[i - 1] * np.exp((k - g) * dt)

    return t, V


def fragility_function(x_bar, sigma, alpha):
    """Fragility relative to even dosing."""
    _, V_even = simulate_tumor(x_bar, 0.0, alpha)
    _, V_uneven = simulate_tumor(x_bar, sigma, alpha)
    return (V_uneven[-1] - V_even[-1]) / V0


# ---------------- Compute fragility curves ---------------- #
fragility_results = {}

for frac in sigma_fractions:
    fragility_curve = []
    for x_bar in dose_values:
        sigma = frac * x_bar
        F = fragility_function(x_bar, sigma, alpha)
        fragility_curve.append(F)
    fragility_results[frac] = np.array(fragility_curve)


# ---------------- Plot ---------------- #
plt.figure(figsize=(7, 5))

for frac, fragility_curve in fragility_results.items():
    if frac == 0:
        label = r"$\sigma = 0$"
    else:
        label = rf"$\sigma = {frac}\bar{{x}}$"
    plt.plot(dose_values, fragility_curve, lw=2, label=label)

plt.axhline(0, color="gray", linestyle="--", linewidth=1)
plt.xlabel(r"Mean dose $\bar{x}$ (mg/L)")
plt.ylabel(r"Fragility $F(\bar{x},\sigma)$")
plt.title(r"Fragility under fast PK for varying schedule variability")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig("misc_analysis/graphs/fragility_vs_sigma_appendix.png", dpi=300)
plt.show()
