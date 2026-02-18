import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import string

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

mean_dose = 20

# PD (Hill kill)
n = 5
C = 25
K_kill = 2.5

# tumour intrinsic growth
k = 0.25

plt.rcParams["axes.titlepad"] = 6  # tighter titles


# ---------------- Helper functions ---------------- #
def gamma_kill(x):
    # Hill kill function γ(x)
    x = np.maximum(x, 0.0)  # safety
    return K_kill * (x**n) / (x**n + C**n)


def pk_profile(t, x_bar, sigma, alpha):
    """Fast-decay, two-dose profile: first half uses x_bar+sigma, second half uses x_bar-sigma."""
    x_t = np.zeros_like(t)
    for i, ti in enumerate(t):
        if ti < T / 2:
            x_t[i] = (x_bar + sigma) * np.exp(-alpha * ti)
        else:
            x_t[i] = (x_bar - sigma) * np.exp(-alpha * (ti - T / 2))
    return x_t


def simulate_tumor(x_bar, sigma, alpha):
    t = np.arange(0, T + dt, dt)
    x_t = pk_profile(t, x_bar, sigma, alpha)

    V = np.zeros_like(t)
    V[0] = V0

    # dV/dt = (k - γ(x(t))) V
    for i in range(1, len(t)):
        g = gamma_kill(x_t[i - 1])
        V[i] = V[i - 1] * np.exp((k - g) * dt)  # stable exact step

    return t, x_t, V


def fragility_function(x_bar, sigma, alpha):
    _, _, V_even = simulate_tumor(x_bar, 0, alpha)
    _, _, V_uneven = simulate_tumor(x_bar, sigma, alpha)
    return (V_uneven[-1] - V_even[-1]) / V0


def compute_kill_curve():
    x_vals = dose_values
    return x_vals, gamma_kill(x_vals)


def set_panel_title(ax, label, title):
    ax.set_title(rf"$\mathbf{{({label})}}$ {title}", loc="left", fontsize=12)


# ---------------- Plot ---------------- #
fig = plt.figure(figsize=(20, 12))
gs_main = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.3)

panel_labels = list(string.ascii_uppercase)
label_idx = 0

for row, alpha in enumerate(alpha_values):
    # Example trajectories
    t_even, x_even, v_even = simulate_tumor(mean_dose, 0, alpha)
    t_uneven, x_uneven, v_uneven = simulate_tumor(
        mean_dose, mean_dose / 2, alpha
    )  # choose uneven sigma from your list

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
    ax0.set_ylabel("Tumour volume (cm³)")
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

    # ---- Column 2: Fragility ---- #
    ax2 = fig.add_subplot(gs_main[row, 2])
    set_panel_title(ax2, panel_labels[label_idx], f"Fragility vs Dose (α={alpha})")
    label_idx += 1
    for sigma in sigma_values:
        ax2.plot(dose_values, fragility_results[sigma], lw=2, label=f"σ={sigma}")
    ax2.axhline(0, color="gray", linestyle="--")
    ax2.set_xlabel("Mean dose x̄ (mg/L)")
    ax2.set_ylabel("Fragility F(x̄, σ)")
    ax2.legend()
    ax2.grid(True)

plt.savefig(
    "misc_analysis/graphs/Fast_PK_Results_labelled.png", dpi=300, bbox_inches="tight"
)
plt.show()


# ---------------- Helper functions ---------------- #
def gamma_kill(x):
    # Hill kill function γ(x)
    x = np.maximum(x, 0.0)  # safety
    return K_kill * (x**n) / (x**n + C**n)


def pk_profile(t, x_bar, sigma, alpha):
    """Fast-decay, two-dose profile: first half uses x_bar+sigma, second half uses x_bar-sigma."""
    x_t = np.zeros_like(t)
    for i, ti in enumerate(t):
        if ti < T / 2:
            x_t[i] = (x_bar + sigma) * np.exp(-alpha * ti)
        else:
            x_t[i] = (x_bar - sigma) * np.exp(-alpha * (ti - T / 2))
    return x_t


def simulate_tumor(x_bar, sigma, alpha):
    t = np.arange(0, T + dt, dt)
    x_t = pk_profile(t, x_bar, sigma, alpha)

    V = np.zeros_like(t)
    V[0] = V0

    # dV/dt = (k - γ(x(t))) V
    for i in range(1, len(t)):
        g = gamma_kill(x_t[i - 1])
        V[i] = V[i - 1] * np.exp((k - g) * dt)  # stable exact step

    return t, x_t, V


def fragility_function(x_bar, sigma, alpha):
    _, _, V_even = simulate_tumor(x_bar, 0, alpha)
    _, _, V_uneven = simulate_tumor(x_bar, sigma, alpha)
    return (V_uneven[-1] - V_even[-1]) / V0


def compute_kill_curve():
    x_vals = dose_values
    return x_vals, gamma_kill(x_vals)


def set_panel_title(ax, label, title):
    ax.set_title(rf"$\mathbf{{({label})}}$ {title}", loc="left", fontsize=12)


# ---------------- Plot ---------------- #
fig = plt.figure(figsize=(20, 12))
gs_main = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.3)

panel_labels = list(string.ascii_uppercase)
label_idx = 0

for row, alpha in enumerate(alpha_values):
    # Example trajectories
    t_even, x_even, v_even = simulate_tumor(mean_dose, 0, alpha)
    t_uneven, x_uneven, v_uneven = simulate_tumor(
        mean_dose, mean_dose / 2, alpha
    )  # choose uneven sigma from your list

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
    ax0.set_ylabel("Tumour volume (cm³)")
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

    # ---- Column 2: Fragility ---- #
    ax2 = fig.add_subplot(gs_main[row, 2])
    set_panel_title(ax2, panel_labels[label_idx], f"Fragility vs Dose (α={alpha})")
    label_idx += 1
    for sigma in sigma_values:
        ax2.plot(dose_values, fragility_results[sigma], lw=2, label=f"σ={sigma}")
    ax2.axhline(0, color="gray", linestyle="--")
    ax2.set_xlabel("Mean dose x̄ (mg/L)")
    ax2.set_ylabel("Fragility F(x̄, σ)")
    ax2.legend()
    ax2.grid(True)

plt.savefig(
    "misc_analysis/graphs/Fast_PK_Results_labelled.png", dpi=300, bbox_inches="tight"
)
plt.show()
