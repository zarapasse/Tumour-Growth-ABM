"""This script runs several iterations of the tumour growth model, computes the mean and standard deviation, and compares the results to the analytic exponential growth solution."""

from src.Model import TumourModel
import matplotlib.pyplot as plt
import numpy as np


def main():
    
    # Define Parameters - Complete guesses for rates for now
    initial_cells = 100
    birth_rate = 0.2  # per time unit
    death_rate = 0.1  # per time unit
    dt = 0.1  # time step
    steps = 200
    T = steps * dt  # total time
    n_runs = 10

    all_populations = np.zeros((n_runs, steps))

    for run in range(n_runs):
        model = TumourModel(initial_cells, birth_rate, death_rate, dt)
        population_sizes = []
        for step in range(steps):
            model.step()
            population_sizes.append(len(model.agents))
        all_populations[run, :] = population_sizes

    # Compute mean and standard deviation across runs
    mean_pop = all_populations.mean(axis=0)
    std_pop = all_populations.std(axis=0)

    # Analytic exponential growth for comparison
    time = np.arange(steps) * dt
    analytic = initial_cells * np.exp((birth_rate - death_rate) * time)

    # Plot results
    plt.figure(figsize=(8, 5))
    plt.plot(time, mean_pop, "b", label="ABM Mean")
    plt.fill_between(
        time,
        mean_pop - std_pop,
        mean_pop + std_pop,
        color="blue",
        alpha=0.2,
        label="±1 Std Dev",
    )
    plt.plot(time, analytic, "k--", label="Analytic Exponential")
    plt.xlabel("Time")
    plt.ylabel("Number of cells")
    plt.title(f"Tumour Cell Population: {n_runs} Simulations vs Analytic Growth")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()