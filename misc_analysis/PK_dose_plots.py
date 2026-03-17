import numpy as np
import matplotlib.pyplot as plt


def pk_concentration_series(t, schedule, alpha):
    conc = np.zeros_like(t, dtype=float)
    for amount, t_dose in schedule:
        mask = t >= t_dose
        conc[mask] += amount * np.exp(-alpha * (t[mask] - t_dose))
    return conc


def _dose_times(*schedules):
    return sorted({td for sch in schedules for _, td in sch})


def figure_clearance_vs_accumulation(
    schedule,
    alpha_low,
    alpha_high,
    t_end,
    n_points=4000,
    title_left_labels=("A", "B"),
    savepath=None,
):
    t = np.linspace(0.0, t_end, n_points)
    c_low = pk_concentration_series(t, schedule, alpha_low)
    c_high = pk_concentration_series(t, schedule, alpha_high)

    ymax = 1.05 * max(c_low.max(), c_high.max())
    dts = _dose_times(schedule)

    fig, axes = plt.subplots(
        2, 1, figsize=(5.2, 4.8), sharex=True, constrained_layout=True
    )

    # Top: low alpha
    ax = axes[0]
    ax.plot(t, c_low, lw=2, color="black")
    ax.fill_between(t, 0, c_low, alpha=0.25, color="black")
    for td in dts:
        ax.axvline(td, ls="--", lw=1, alpha=0.5, color="black")
    ax.set_ylabel("Conc. (mg/L)")
    ax.set_ylim(0, ymax)
    ax.grid(True, alpha=0.3)
    ax.set_title(
        rf"$\mathbf{{({title_left_labels[0]})}}$ Drug Concentration ($\alpha$={alpha_low})",
        loc="left",
        fontsize=10,
    )

    # Bottom: high alpha
    ax = axes[1]
    ax.plot(t, c_high, lw=2, color="black")
    ax.fill_between(t, 0, c_high, alpha=0.25, color="black")
    for td in dts:
        ax.axvline(td, ls="--", lw=1, alpha=0.5, color="black")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Conc. (mg/L)")
    ax.set_ylim(0, ymax)
    ax.grid(True, alpha=0.3)
    ax.set_title(
        rf"$\mathbf{{({title_left_labels[1]})}}$ Drug Concentration ($\alpha$={alpha_high})",
        loc="left",
        fontsize=10,
    )

    if savepath:
        plt.savefig(savepath, dpi=300)
    # plt.show()


def figure_even_vs_uneven_side_by_side_alphas(
    even_schedule,
    uneven_schedule,
    alpha_low,
    alpha_high,
    t_end,
    n_points=4000,
    left_labels=(("A", "B"), ("C", "D")),  # (row1 col1,col2), (row2 col1,col2)
    savepath=None,
):
    t = np.linspace(0.0, t_end, n_points)
    dts = _dose_times(even_schedule, uneven_schedule)

    # Precompute for consistent y-limits across all 4 panels
    c_e_low = pk_concentration_series(t, even_schedule, alpha_low)
    c_e_high = pk_concentration_series(t, even_schedule, alpha_high)
    c_u_low = pk_concentration_series(t, uneven_schedule, alpha_low)
    c_u_high = pk_concentration_series(t, uneven_schedule, alpha_high)
    ymax = 1.05 * max(c_e_low.max(), c_e_high.max(), c_u_low.max(), c_u_high.max())

    fig, axes = plt.subplots(
        2, 2, figsize=(10.4, 4.8), sharex=True, sharey=True, constrained_layout=True
    )

    # ---- Top row: EVEN ----
    ax = axes[0, 0]
    ax.plot(t, c_e_low, lw=2, label="Even", color="black")
    ax.fill_between(t, 0, c_e_low, alpha=0.25, color="black")
    for td in dts:
        ax.axvline(td, ls="--", lw=1, alpha=0.5, color="black")
    ax.grid(True, alpha=0.3)
    ax.set_title(
        rf"$\mathbf{{({left_labels[0][0]})}}$ Even ($\alpha$={alpha_low})",
        loc="left",
        fontsize=10,
    )
    ax.set_ylabel("Conc. (mg/L)")
    ax.set_ylim(0, ymax)
    ax.legend(loc="upper right", fontsize=8, frameon=True)

    ax = axes[0, 1]
    ax.plot(t, c_e_high, lw=2, label="Even", color="black")
    ax.fill_between(t, 0, c_e_high, alpha=0.25, color="black")
    for td in dts:
        ax.axvline(td, ls="--", lw=1, alpha=0.5, color="black")
    ax.grid(True, alpha=0.3)
    ax.set_title(
        rf"$\mathbf{{({left_labels[0][1]})}}$ Even ($\alpha$={alpha_high})",
        loc="left",
        fontsize=10,
    )
    ax.set_ylim(0, ymax)
    ax.legend(loc="upper right", fontsize=8, frameon=True)

    # ---- Bottom row: UNEVEN ----
    ax = axes[1, 0]
    ax.plot(t, c_u_low, lw=2, label="Uneven", color="black")
    ax.fill_between(t, 0, c_u_low, alpha=0.25, color="black")
    for td in dts:
        ax.axvline(td, ls="--", lw=1, alpha=0.5, color="black")
    ax.grid(True, alpha=0.3)
    ax.set_title(
        rf"$\mathbf{{({left_labels[1][0]})}}$ Uneven ($\alpha$={alpha_low})",
        loc="left",
        fontsize=10,
    )
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Conc. (mg/L)")
    ax.set_ylim(0, ymax)
    ax.legend(loc="upper right", fontsize=8, frameon=True)

    ax = axes[1, 1]
    ax.plot(t, c_u_high, lw=2, label="Uneven", color="black")
    ax.fill_between(t, 0, c_u_high, alpha=0.25, color="black")
    for td in dts:
        ax.axvline(td, ls="--", lw=1, alpha=0.5, color="black")
    ax.grid(True, alpha=0.3)
    ax.set_title(
        rf"$\mathbf{{({left_labels[1][1]})}}$ Uneven ($\alpha$={alpha_high})",
        loc="left",
        fontsize=10,
    )
    ax.set_xlabel("Time (days)")
    ax.set_ylim(0, ymax)
    ax.legend(loc="upper right", fontsize=8, frameon=True)

    for ax in axes.ravel():
        ax.set_xlim(0.0, t_end)

    if savepath:
        plt.savefig(savepath, dpi=300)
    # plt.show()


def figure_one_PK_only(
    schedule,
    alpha,
    t_end,
    n_points=4000,
    title_left_label="A",
    schedule_label=None,
    savepath=None,
):
    t = np.linspace(0.0, t_end, n_points)
    c = pk_concentration_series(t, schedule, alpha)

    dts = _dose_times(schedule)

    # Colour selection
    color_map = {
        "Even": "tab:orange",
        "Uneven": "tab:green",
    }
    color = color_map.get(schedule_label, "black")

    fig, ax = plt.subplots(figsize=(5.2, 3.8), constrained_layout=True)

    ax.plot(t, c, lw=2, color=color)
    ax.fill_between(t, 0, c, alpha=0.25, color=color)

    # for td in dts:
    #     ax.axvline(td, ls="--", lw=1, alpha=0.5, color=color)

    ax.set_xlim(0.0, t_end)
    ax.set_ylim(0, 1.05 * c.max())

    ax.set_xlabel("Time")
    ax.set_ylabel("Drug Concentration")
    ax.grid(True, alpha=0.3)

    label_text = f"{schedule_label} " if schedule_label else ""

    ax.set_title(
        rf"{label_text}Dosing Schedule",
        fontsize=10,
    )

    if savepath:
        plt.savefig(savepath, dpi=300)

    plt.show()


# Same timing and dose amounts
schedule = [(20.0, 0.0), (20.0, 1.0)]
figure_clearance_vs_accumulation(
    schedule,
    alpha_low=1,
    alpha_high=4,
    t_end=2.0,
    savepath="misc_analysis/graphs/PK_Clearance_vs_Accumulation.png",
)


even_schedule = [(20.0, 0.0), (20.0, 1.0)]  # total 40
uneven_schedule = [(30.0, 0.0), (10.0, 1.0)]  # total 40, uneven split

figure_even_vs_uneven_side_by_side_alphas(
    even_schedule,
    uneven_schedule,
    alpha_low=1,
    alpha_high=4,
    t_end=2.0,
    savepath="misc_analysis/graphs/PK_Even_vs_Uneven_Side_by_Side.png",
)

figure_one_PK_only(
    even_schedule,
    alpha=5,
    t_end=2.0,
    schedule_label="Even",
    savepath="misc_analysis/graphs/PK_even.png",
)

figure_one_PK_only(
    uneven_schedule,
    alpha=5,
    t_end=2.0,
    schedule_label="Uneven",
    savepath="misc_analysis/graphs/PK_uneven.png",
)
