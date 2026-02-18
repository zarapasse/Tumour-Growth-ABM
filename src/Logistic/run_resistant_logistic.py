import json
import copy
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from utils_logistic import (
    make_fragility_test_scenarios,
    process_config,
    run_abm_resistant_logistic,
    continuum_logistic_with_pkpd,
    estimate_K_tail,
    fit_r_logit,
)

# ---------------- Load Config ---------------- #
CONFIG_PATH = Path(__file__).parent / "config_logistic.json"
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

# ---------------- Build scenarios ---------------- #
x_bar = 20
n_doses_per_cycle = 2
total_dose_per_cycle = x_bar * n_doses_per_cycle
sigma = x_bar / 2
n_cycles = 4
cycle_length = 12
alpha_val = 1

scenarios = make_fragility_test_scenarios(
    total_dose_per_cycle,
    n_doses_per_cycle,
    n_cycles,
    sigma,
    cycle_length,
    alpha_val,
)

# ---------------------- Run ABM ---------------------- #
(
    initial_cells,
    birth_rate,
    death_rate,
    dt,
    steps,
    n_runs,
    hill_params,
    res_params,
    initial_resources,
    initial_cell_energy,
    p_mutation,
    initial_resistant_fraction,
    fitness_cost,
) = process_config(config)

abm_results = run_abm_resistant_logistic(config, scenarios, seed=42)
time = abm_results["time"]

# ---------------------- Run ABM (no-drug baseline for r,K) ---------------------- #
print("\nComputing NO-DRUG baseline (for K)...")

baseline_scenario = [{"name": "No Drug", "schedule": [], "alpha": 0.0}]
cfg = copy.deepcopy(config)
cfg["simulation"]["n_runs"] = 10

# ---- 1) Estimate K from saturated no-drug run ----
K_fit_results = run_abm_resistant_logistic(cfg, baseline_scenario, seed=42)
mean_no_drug = K_fit_results["mean_total"][0]

K0 = estimate_K_tail(mean_no_drug, frac_tail=0.2)

baseline = {
    "mean_total": K_fit_results["mean_total"][0],
    "std_total": K_fit_results["std_total"][0],
}

# ---- 2) Estimate r from low-N no-drug run ----
cfg["simulation"]["initial_cells"] = 30

r_fit_results = run_abm_resistant_logistic(cfg, baseline_scenario, seed=42)

mean_lowN = r_fit_results["mean_total"][0]
time_lowN = r_fit_results["time"]

r, _ = fit_r_logit(time_lowN, mean_lowN, K0, low_frac=0.2, high_frac=0.8)
print("K0 (saturated run):", K0)
print("r0 (low-N run):", r)

# ---------------------- Build deterministic + collect results ---------------------- #
results = []
for i, sc in enumerate(scenarios):
    mean_abm = abm_results["mean_total"][i]
    std_abm = abm_results["std_total"][i]

    N_det, conc_det, _, _ = continuum_logistic_with_pkpd(
        time=time,
        N0=mean_abm[0],
        K=K0,
        r=r,
        schedule=sc["schedule"],
        alpha=sc["alpha"],
        hill_params=hill_params,
    )

    results.append(
        {
            "name": sc["name"],
            "schedule": sc["schedule"],
            "mean_total": abm_results["mean_total"][i],
            "std_total": abm_results["std_total"][i],
            "mean_sensitive": abm_results["mean_sensitive"][i],
            "std_sensitive": abm_results["std_sensitive"][i],
            "mean_resistant": abm_results["mean_resistant"][i],
            "std_resistant": abm_results["std_resistant"][i],
            "N_det": N_det,
            "conc_det": conc_det,
        }
    )


# ------------------- Layout ------------------- #
fig = plt.figure(figsize=(14, 12))
gs = gridspec.GridSpec(3, 2, height_ratios=[2.2, 1.6, 1.2])

ax_top = fig.add_subplot(gs[0, :])
ax_even = fig.add_subplot(gs[1, 0])
ax_odd = fig.add_subplot(gs[1, 1])
ax_pk = fig.add_subplot(gs[2, :])

# ------------------- Colours ------------------- #
col_even = "tab:orange"
col_odd = "tab:green"
col_base = "0.4"  # grey

# ------------------- TOP: total tumour trajectories ------------------- #
for res, col in zip(results, [col_even, col_odd]):

    mean_total = res["mean_total"]
    std_total = res["std_total"]

    ax_top.plot(time, mean_total, color=col, lw=2, label=res["name"])
    ax_top.fill_between(
        time,
        mean_total - std_total,
        mean_total + std_total,
        color=col,
        alpha=0.15,
    )

    # deterministic overlay
    ax_top.plot(time, res["N_det"], ls="--", lw=2, color=col, alpha=0.5)

    # dose markers (optional but nice)
    for _, t_dose in res["schedule"]:
        ax_top.axvline(t_dose, color=col, ls=":", alpha=0.18)

# ---- baseline ONCE ----
if baseline is not None:
    ax_top.plot(
        time,
        baseline["mean_total"],
        "--",
        color=col_base,
        lw=2,
        label="No-drug baseline",
    )
    ax_top.fill_between(
        time,
        baseline["mean_total"] - baseline["std_total"],
        baseline["mean_total"] + baseline["std_total"],
        color=col_base,
        alpha=0.10,
    )

ax_top.set_ylabel("Cells")
ax_top.set_title("Tumour population (total): ABM (±1 SD) and deterministic overlay")
ax_top.legend()
ax_top.grid(alpha=0.3)


# ------------------- MIDDLE: composition stackplots ------------------- #
def plot_composition(ax, res, title):
    sens = res["mean_sensitive"]
    resi = res["mean_resistant"]

    ax.stackplot(
        time,
        sens,
        resi,
        labels=["Sensitive", "Resistant"],
        colors=["tab:blue", "tab:red"],
        alpha=0.85,
    )
    ax.set_title(title)
    ax.set_ylabel("Cells")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)


plot_composition(ax_even, results[0], "Even schedule: Sensitive vs Resistant")
plot_composition(ax_odd, results[1], "Odd schedule: Sensitive vs Resistant")

# ------------------- BOTTOM: PK profiles ------------------- #
ax_pk.plot(time, results[0]["conc_det"], color=col_even, lw=2, label="Even schedule")
ax_pk.plot(time, results[1]["conc_det"], color=col_odd, lw=2, label="Odd schedule")

ax_pk.set_xlabel("Time (days)")
ax_pk.set_ylabel("Drug concentration")
ax_pk.set_title("PK profiles")
ax_pk.legend()
ax_pk.grid(alpha=0.3)

# ----------------- parameter info box ----------------- #
param_lines = [
    "Simulation",
    "----------",
    f"initial_cells = {initial_cells}",
    f"birth_rate    = {birth_rate}",
    f"death_rate    = {death_rate}",
    f"dt            = {dt}",
    f"steps         = {steps}",
    f"n_runs        = {n_runs}",
    "",
    "Resources",
    "---------",
    f"energy_capacity  = {res_params.energy_capacity}",
    f"div_threshold    = {res_params.division_threshold}",
    f"maintenance_cost = {res_params.maintenance_cost}",
    f"resource_influx  = {res_params.resource_influx}",
    "",
    "Resistance",
    "----------",
    f"p_mutation              = {p_mutation}",
    f"initial_resistant_frac  = {initial_resistant_fraction}",
    f"fitness_cost            = {fitness_cost}",
    "",
    "Hill PD",
    "-------",
    f"K_kill = {hill_params.K_kill}",
    f"C      = {hill_params.C}",
    f"n      = {hill_params.n}",
]

param_text = "\n".join(param_lines)


# ------------------- Panel Labels ------------------- #
ax_top.text(
    0.0,
    1.02,
    r"$\mathbf{(A)}$",
    transform=ax_top.transAxes,
    ha="left",
    va="bottom",
)

ax_even.text(
    0.0,
    1.02,
    r"$\mathbf{(B)}$",
    transform=ax_even.transAxes,
    ha="left",
    va="bottom",
)

ax_odd.text(
    0.0,
    1.02,
    r"$\mathbf{(C)}$",
    transform=ax_odd.transAxes,
    ha="left",
    va="bottom",
)

ax_pk.text(
    0.0,
    1.02,
    r"$\mathbf{(D)}$",
    transform=ax_pk.transAxes,
    ha="left",
    va="bottom",
)

plt.tight_layout(rect=(0, 0, 0.88, 1.0))
fig.text(
    0.885,
    0.98,
    param_text,
    fontsize=7,
    va="top",
    ha="left",
    family="monospace",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="0.8"),
)

out_path = (
    Path(__file__).parent
    / "Graphs"
    / "Resistance"
    / f"logistic_abm_trajectories_{initial_cells}_cells_{n_runs}_runs_{x_bar}.png"
)

out_path.parent.mkdir(parents=True, exist_ok=True)

plt.savefig(out_path, dpi=300)
plt.show()
# Close figure when running on command line
# plt.close()
