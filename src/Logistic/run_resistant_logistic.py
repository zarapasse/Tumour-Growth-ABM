import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import json
from pathlib import Path


from utils_logistic import (
    make_fragility_test_scenarios,
    process_scenarios_config,
    process_config,
    deterministic_pk,
    run_abm_logistic_resistant,
)


# ---------------- Load Config ---------------- #
CONFIG_PATH = Path(__file__).parent / "config_logistic.json"
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

do_fragility_test = True


# ---------------- Build scenarios ---------------- #
if do_fragility_test:
    print("Running fragility test for logistic ABM...")

    x_bar = 40
    n_doses_per_cycle = 2
    total_dose_per_cycle = x_bar * n_doses_per_cycle
    sigma = total_dose_per_cycle / 2
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
    is_fragility = True

else:
    scenarios = process_scenarios_config(config)
    is_fragility = False


# Extract config parameters for baseline computation
initial_cells, birth_rate, death_rate, dt, steps, n_runs, hill_params = process_config(
    config
)
time = np.arange(steps + 1) * dt


# ---------------- Compute NO-DRUG baseline ---------------- #
print("\nComputing NO-DRUG baseline...")

baseline_scenario = [
    {
        "name": "No Drug",
        "schedule": [],
        "alpha": 0.0,
    }
]

baseline_results = run_abm_logistic_resistant(config, baseline_scenario, seed=42)

mean_no_drug = baseline_results["mean_trajectories"][0]
std_no_drug = baseline_results["std_trajectories"][0]


# ---------------- Run ABM ---------------- #
initial_cells, birth_rate, death_rate, dt, steps, n_runs, hill_params = process_config(
    config
)
time = np.arange(steps + 1) * dt


abm_results = run_abm_logistic_resistant(
    config, scenarios, seed=42, compute_fragility=True
)

# ---------------- Collect results ---------------- #
results = []

for i, sc in enumerate(scenarios):

    mean_abm = abm_results["mean_trajectories"][i]
    std_abm = abm_results["std_trajectories"][i]

    # deterministic PK curve
    conc_det = deterministic_pk(time, sc["schedule"], sc["alpha"])

    results.append(
        {
            "name": sc["name"],
            "schedule": sc["schedule"],
            "alpha": sc["alpha"],
            "mean": mean_abm,
            "std": std_abm,
            "conc_det": conc_det,
        }
    )


# ------------------- Plotting ------------------- #
scenario_palette = [
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
    "tab:olive",
    "tab:cyan",
]

fig = plt.figure(figsize=(14, 12))
gs = gridspec.GridSpec(3, 2, height_ratios=[2.2, 1.6, 1.2])

ax_top = fig.add_subplot(gs[0, :])
ax_even = fig.add_subplot(gs[1, 0])
ax_odd  = fig.add_subplot(gs[1, 1])
ax_pk   = fig.add_subplot(gs[2, :])

# ===================== TOP: Tumour Trajectories =====================
mean_no_drug_total = mean_no_drug[:, 0] + mean_no_drug[:, 1]


for res, col in zip(results, ["tab:orange", "tab:green"]):
    mean_total = res["mean"][:,0] + res["mean"][:,1]
    std_total  = np.sqrt(res["std"][:,0]**2 + res["std"][:,1]**2)

    ax_top.plot(time, mean_total, color=col, lw=2, label=res["name"])
    ax_top.fill_between(time,
                        mean_total-std_total,
                        mean_total+std_total,
                        color=col, alpha=0.15)

ax_top.plot(time, mean_no_drug[:,0] + mean_no_drug[:,1],
            "--", color="grey", lw=2, label="No-drug baseline")

ax_top.set_ylabel("Cells")
ax_top.set_title("Tumour population: ABM vs no-drug baseline")
ax_top.legend()
ax_top.grid(alpha=0.3)


def plot_composition(ax, res, title):
    sensitive = res["mean"][:,0]
    resistant = res["mean"][:,1]

    ax.stackplot(
        time,
        sensitive,
        resistant,
        labels=["Sensitive", "Resistant"],
        colors=["tab:blue", "tab:red"],
        alpha=0.8
    )
    ax.set_title(title)
    ax.set_ylabel("Cells")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)

plot_composition(ax_even, results[0], "Even schedule: Sensitive vs Resistant")
plot_composition(ax_odd,  results[1], "Odd schedule: Sensitive vs Resistant")









# ===================== BOTTOM: PK Profiles =====================
ax_pk.plot(time, results[0]["conc_det"], color="tab:orange", lw=2,
           label="Even schedule")
ax_pk.plot(time, results[1]["conc_det"], color="tab:green", lw=2,
           label="Odd schedule")

ax_pk.set_xlabel("Time")
ax_pk.set_ylabel("Drug concentration")
ax_pk.set_title("PK profiles")
ax_pk.legend()
ax_pk.grid(alpha=0.3)
# ----------------- parameter info box ----------------- #
param_lines = [
    f"initial_cells: {initial_cells}",
    f"birth_rate: {birth_rate}",
    f"death_rate: {death_rate}",
    f"dt: {dt}, steps: {steps}",
    f"n_runs: {n_runs}",
    "Hill parameters:",
    f"  E0: {hill_params['E0']}",
    f"  E1: {hill_params['E1']}",
    f"  C:  {hill_params['C']}",
    f"  n:  {hill_params['n']}",
]
param_text = "\n".join(param_lines)

# reserve a narrow area on the right for the slim info box and draw the text
plt.tight_layout(
    rect=(0, 0, 0.88, 1.0)
)  # leave ~12% on the right for the info box (moved left)
fig.text(
    0.855,  # moved left so the box sits just next to the axes
    0.98,  # start from top so lines flow downward
    param_text,
    fontsize=7,  # smaller font to fit the slim box
    va="top",
    ha="left",
    family="monospace",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="0.8"),
)


plt.tight_layout()
plt.show()


# ------------------------------------------------------
# FRAGILITY PRINT
# ------------------------------------------------------
if is_fragility:
    frag = abm_results["fragility"]
    print("\n===== FRAGILITY ANALYSIS (LOGISTIC ABM) =====")
    print(f"Mean fragility = {frag['mean']:.4f}")
    print(f"Per-run fragility = {frag['per_run']}")
