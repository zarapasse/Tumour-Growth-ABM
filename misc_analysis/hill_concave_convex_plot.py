import numpy as np
import matplotlib.pyplot as plt

# --- Parameters ---
K_kill = 1.0     # maximum kill rate
C = 1             # EC50 / C in your notation
n = 5.0        # Hill coefficient (choose >1 to see convex->concave change)

# --- Hill function ---
def gamma(x):
    return K_kill * (x**n) / (C**n + x**n)

# Inflection point (only for n>1)
xB = C * ((n - 1) / (n + 1))**(1 / n) if n > 1 else 0.0

# --- Grid ---
x_max = 2 * C
x = np.linspace(1e-6, x_max, 2000)
y = gamma(x)

# --- Plot ---
plt.figure(figsize=(7, 4.5))
plt.plot(
    x, y, linewidth=2,
    label=rf"$\gamma(x)$ (n={n:g})"
)

# Shade convex (antifragile) and concave (fragile) regions
if n > 1:
    left = x <= xB
    right = x >= xB
    plt.fill_between(x[left], y[left], alpha=0.2, label="convex (antifragile)")
    plt.fill_between(x[right], y[right], alpha=0.2, label="concave (fragile)")
    plt.axvline(xB, linestyle="--", linewidth=1.5)
    plt.text(xB, 0.05*K_kill, r"$\bar{x}_B$", rotation=90, va="bottom", ha="right")
else:
    # For n=1, curve is everywhere concave; shade all as fragile
    plt.fill_between(x, y, alpha=0.2, label="concave (fragile)")

plt.xlabel("drug concentration (mg/L)")
plt.ylabel(r"$\gamma(x)$")
plt.ylim(-0.02*K_kill, 1.05*K_kill)
plt.xlim(0, x_max)
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig("misc_analysis/graphs/Hill_Concave_Convex_Plot.png", dpi=300)
plt.show()