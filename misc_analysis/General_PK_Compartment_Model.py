# No cap general PK compartment model simulation and analysis
# Panel labels are embedded in titles as bold (A), (B), ... with normal-weight title text.
# Colours are consistent: Even = blue (C0), Uneven = orange (C1).
# Units: time in days, volume in cm^3, concentration/dose in mg/L, DIP in day^{-1}.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import string

# ---------------- Parameters ---------------- #
T = 2
alpha_values = [0.1, 0.3, 0.5]  # alpha values to compare
dose_values = np.linspace(0, 50, 200)
sigma_values = [7, 9, 11, 13, 15]
dt = 0.01

V0 = 0.001  # cm^3 (so 0.001 cm^3 = 1 mm^3)
n = 10
C = 20
E0 = 1
E1 = -1

plt.rcParams["axes.titlepad"] = 6  # slightly tighter title padding

# ---------------- Helper functions ---------------- #
def hill_effect(x, n, C):
    return E0 + (x**n * (E1 - E0)) / (x**n + C**n)

def simulate_tumor(x_bar, sigma, alpha):
    """
    General PK (accumulation): first dose always contributes; second dose adds after T/2.
    """
    t = np.arange(0, T, dt)
    x_t = np.zeros_like(t)

    for i, ti in enumerate(t):
        dose1_contrib = (x_bar + sigma) * np.exp(-alpha * ti)
        if ti >= T / 2:
            dose2_contrib = (x_bar - sigma) * np.exp(-alpha * (ti - T / 2))
            x_t[i] = dose1_contrib + dose2_contrib
        else:
            x_t[i] = dose1_contrib

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
    """
    Visual DIP curve at a fixed time point t=0.1 (as in your previous script).
    """
    x_vals = dose_values * np.exp(-alpha * 0.1)
    return dose_values, hill_effect(x_vals, n, C)

def set_panel_title(ax, label, title):
    # Bold label only, normal title text
    ax.set_title(rf"$\mathbf{{({label})}}$ {title}", loc="left", fontsize=12)

# ---------------- Plot ---------------- #
fig = plt.figure(figsize=(20, 12))
gs_main = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.3)

panel_labels = list(string.ascii_uppercase)
label_idx = 0

for row, alpha in enumerate(alpha_values):

    # Compute tumour trajectory and drug concentrations
    t_even, x_even, v_even = simulate_tumor(x_bar=25, sigma=0, alpha=alpha)
    t_uneven, x_uneven, v_uneven = simulate_tumor(x_bar=25, sigma=25, alpha=alpha)

    # Compute fragility analysis
    fragility_results = {}
    for sigma in sigma_values:
        fragility_results[sigma] = np.array(
            [fragility_function(x_bar, sigma, alpha) for x_bar in dose_values]
        )

    # ---- Column 0: Tumour trajectory ---- #
    ax0 = fig.add_subplot(gs_main[row, 0])
    set_panel_title(ax0, panel_labels[label_idx], f"Tumour Trajectory (α={alpha})")
    label_idx += 1

    ax0.plot(t_even, v_even, lw=2, color="C0", label="Even dosing")
    ax0.plot(t_uneven, v_uneven, lw=2, color="C1", label="Uneven dosing")
    ax0.set_xlabel("Time (days)")
    ax0.set_ylabel(r"Tumour volume (cm$^3$)")
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
    ax1a.plot(t_even, x_even, lw=2, color="C0", label="Even")
    ax1a.axvline(T / 2, color="gray", linestyle="--", alpha=0.5)
    ax1a.set_ylabel("Conc. (mg/L)")
    ax1a.legend(loc="upper right")
    ax1a.grid(True)
    ax1a.tick_params(labelbottom=False)

    # Uneven dosing (bottom) — ORANGE
    ax1b = fig.add_subplot(gs_drug[1], sharex=ax1a)
    ax1b.fill_between(t_uneven, x_uneven, color="C1", alpha=0.25)
    ax1b.plot(t_uneven, x_uneven, lw=2, color="C1", label="Uneven")
    ax1b.axvline(T / 2, color="gray", linestyle="--", alpha=0.5)
    ax1b.set_xlabel("Time (days)")
    ax1b.set_ylabel("Conc. (mg/L)")
    ax1b.legend(loc="upper right")
    ax1b.grid(True)

    # ---- Column 2: Hill / DIP curve ---- #
    ax2 = fig.add_subplot(gs_main[row, 2])
    set_panel_title(ax2, panel_labels[label_idx], f"DIP Curve (α={alpha}) at t=0.1")
    label_idx += 1

    x_vals, hill_vals = compute_hill_curve(alpha)
    ax2.plot(x_vals, hill_vals, lw=2, color="black")
    ax2.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax2.set_xlabel(r"Drug dose $\bar{x}$ (mg/L)")
    ax2.set_ylabel(r"DIP rate $H(\bar{x})$")
    ax2.grid(True)

    # ---- Column 3: Fragility ---- #
    ax3 = fig.add_subplot(gs_main[row, 3])
    set_panel_title(ax3, panel_labels[label_idx], f"Fragility vs Dose (α={alpha})")
    label_idx += 1

    for sigma in sigma_values:
        ax3.plot(dose_values, fragility_results[sigma], lw=2, label=f"σ={sigma}")
    ax3.axhline(0, color="gray", linestyle="--")
    ax3.set_xlabel(r"Drug dose $\bar{x}$ (mg/L)")
    ax3.set_ylabel(r"Fragility $F(\bar{x},\sigma)$")
    ax3.legend()
    ax3.grid(True)

# Save + show
plt.savefig("misc_analysis/graphs/General_PK_Results_labelled.png", dpi=300, bbox_inches="tight")
print("Figure saved as 'misc_analysis/graphs/General_PK_Result_labelled.png'")
plt.show()