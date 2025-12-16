import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path("data/processed")
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

V0 = 1000  # or load from params

fragility = {}
resistance = {}

for csv in DATA_DIR.glob("even_x*.csv"):
    x_bar = float(csv.stem.split("x")[1])

    df_even = pd.read_csv(csv)
    df_odd = pd.read_csv(DATA_DIR / f"odd_x{x_bar:.0f}.csv")

    V_even = df_even["Total_mean"].iloc[-1]
    V_odd = df_odd["Total_mean"].iloc[-1]

    F = (V_odd - V_even) / V0
    fragility[x_bar] = F

    resistance[x_bar] = df_odd["ResistantFraction_mean"].iloc[-1]

# Sort by dose
x = np.array(sorted(fragility.keys()))
F = np.array([fragility[v] for v in x])
R = np.array([resistance[v] for v in x])

# Plot fragility vs dose
plt.figure()
plt.plot(x, F, "o-", lw=2)
plt.axhline(0, color="grey", ls="--")
plt.xlabel("Mean dose per administration")
plt.ylabel("Fragility")
plt.savefig(FIG_DIR / "fragility_vs_dose.png", dpi=300)
plt.close()