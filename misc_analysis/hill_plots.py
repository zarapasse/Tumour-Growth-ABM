import numpy as np
import matplotlib.pyplot as plt

# Parameters
C = 1.0           # EC50 (half-max concentration)
K_kill = 1.0      # max kill rate

# Concentration range
x = np.linspace(0, 3*C, 600)

def gamma_hill(x, n, C, K_kill):
    return K_kill * (x**n) / (x**n + C**n)

# Hill curves for different n values
n_values = [1, 2, 4, 8]
for n in n_values:
    plt.plot(x, gamma_hill(x, n, C, K_kill), label=rf"$n={n}$")

plt.xlabel(r"Concentration $x$")
plt.ylabel(r"Kill rate $\gamma(x)$")
plt.title(r"Hill kill function for different coefficients $n$")
plt.ylim(0, 1.05*K_kill)
plt.xlim(0, 3*C)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("misc_analysis/graphs/Hill_Kill_Functions.png", dpi=300)
plt.show()