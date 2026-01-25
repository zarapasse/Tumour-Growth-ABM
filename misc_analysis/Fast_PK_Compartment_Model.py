# This script plots the 1-Compartment PK model from Pierek with (α>0) and without (α=0)
# drug decay for FAST drug decay, with panel labels integrated into titles.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import string

# ---------------- Parameters ---------------- #
T = 2
alpha_values = [0, 3, 5]
dose_values = np.linspace(0, 50, 200)
sigma_values = [0, 1, 3, 5, 7]
dt = 0.01
V0 = 0.001
n = 10
C = 20
E0 = 1
E1 = -1

plt.rcParams["axes.titlepad"] = 6  # tighter titles

# ---------------- Helper functions ---------------- #
def hill_effect(x, n, C):
    return E0 + (x**n * (E1 - E0)) / (x**n + C**n)

def simulate_tumor(x_bar, sigma, alpha):
    t = np.arange(0, T, dt)
    x_t = np.zeros_like(t)

    for i, ti in enumerate(t):
        if ti < T / 2:
            x_t[i] = (x_bar + sigma) * np.exp(-alpha * ti)
        else:
            x_t[i] = (x_bar - sigma) * np.exp(-alpha * (ti - T / 2))

    V = np.zeros_like(t)
    V[0] = V0
    for i in range(1, len(t)):
        V[i] = V[i - 1] * np.exp(hill_effect(x_t[i - 1], n, C) * dt)

    return t, x_t, V

def fragility_function(x_bar, sigma, alpha):
    _, _, V_even = simulate_tumor(x_bar, 0, alpha)
    _, _, V_uneven = simulate_tumor(x_bar, sigma, alpha)
    return (V_uneven[-1] - V_even[-1]) / V0

def compute_hill_curve(alpha):
    x_vals = dose_values * np.exp(-alpha * 0.1)
    return dose_values, hill_effect(x_vals, n, C)

def set_panel_title(ax, label, title):
    ax.set_title(
        rf"$\mathbf{{({label})}}$ {title}",
        loc="left",
        fontsize=12
    )
# ---------------- Plot ---------------- #
fig = plt.figure(figsize=(20, 12))
gs_main = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.3)

panel_labels = list(string.ascii_uppercase)
label_idx = 0

for row, alpha in enumerate(alpha_values):

    t_even, x_even, v_even = simulate_tumor(25, 0, alpha)
    t_uneven, x_uneven, v_uneven = simulate_tumor(25, 25, alpha)

    fragility_results = {}
    for sigma in sigma_values:
        fragility_results[sigma] = np.array(
            [fragility_function(x_bar, sigma, alpha) for x_bar in dose_values]
        )

    # ---- Column 0: Tumour trajectory ---- #
    ax0 = fig.add_subplot(gs_main[row, 0])
    set_panel_title(ax0, panel_labels[label_idx], f"Tumour Trajectory (α={alpha})")
    label_idx += 1

    ax0.plot(t_even, v_even, lw=2, label="Even dosing")
    ax0.plot(t_uneven, v_uneven, lw=2, label="Uneven dosing")
    ax0.set_xlabel("Time (days)")
    ax0.set_ylabel("Tumour volume V(t) (cm³)")
    ax0.legend()
    ax0.grid(True)
    # ---- Column 1: Drug concentration (stacked) ---- #
    gs_drug = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs_main[row, 1], hspace=0.05
    )

    # Even dosing (top) — BLUE
    ax1a = fig.add_subplot(gs_drug[0])
    set_panel_title(ax1a, panel_labels[label_idx], f"Drug Concentration (α={alpha})")
    label_idx += 1

    ax1a.fill_between(t_even, x_even, color="C0", alpha=0.25)
    ax1a.plot(t_even, x_even, color="C0", lw=2, label="Even")
    ax1a.axvline(T / 2, color="gray", linestyle="--", alpha=0.5)
    ax1a.set_ylabel("Conc. (mg/L)")
    ax1a.legend()
    ax1a.grid(True)
    ax1a.tick_params(labelbottom=False)

    # Uneven dosing (bottom) — ORANGE
    ax1b = fig.add_subplot(gs_drug[1], sharex=ax1a)
    ax1b.fill_between(t_uneven, x_uneven, color="C1", alpha=0.25)
    ax1b.plot(t_uneven, x_uneven, color="C1", lw=2, label="Uneven")
    ax1b.axvline(T / 2, color="gray", linestyle="--", alpha=0.5)
    ax1b.set_xlabel("Time (days)")
    ax1b.set_ylabel("Conc. (mg/L)")
    ax1b.legend()
    ax1b.grid(True)

    # ---- Column 2: DIP curve ---- #
    ax2 = fig.add_subplot(gs_main[row, 2])
    set_panel_title(ax2, panel_labels[label_idx], f"DIP Curve (α={alpha}) at t=0.1")
    label_idx += 1

    x_vals, hill_vals = compute_hill_curve(alpha)
    ax2.plot(x_vals, hill_vals, lw=2, color="black")
    ax2.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Drug dose x̄ (mg/L)")
    ax2.set_ylabel("H(x̄) (day$^{-1}$)")
    ax2.grid(True)

    # ---- Column 3: Fragility ---- #
    ax3 = fig.add_subplot(gs_main[row, 3])
    set_panel_title(ax3, panel_labels[label_idx], f"Fragility vs Dose (α={alpha})")
    label_idx += 1

    for sigma in sigma_values:
        ax3.plot(dose_values, fragility_results[sigma], lw=2, label=f"σ={sigma}")
    ax3.axhline(0, color="gray", linestyle="--")
    ax3.set_xlabel("Drug dose x̄ (mg/L)")
    ax3.set_ylabel("Fragility F(x̄, σ)")
    ax3.legend()
    ax3.grid(True)

# ---------------- Save ---------------- #
plt.savefig("misc_analysis/graphs/Fast_PK_Results_labelled.png", dpi=300, bbox_inches="tight")
plt.show()