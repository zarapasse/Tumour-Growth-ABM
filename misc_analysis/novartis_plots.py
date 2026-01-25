"""
Exploratory plots for untreated control tumour growth variability.

This script visualises inter-tumour heterogeneity in summary growth metrics
for a single tumour type (CRC), motivating the use of low-dimensional ODE
growth models for volume-only datasets.

Figures produced:
1. Histogram of time to tumour doubling
2. Scatter plot of time to double vs best response
"""

import pandas as pd
import matplotlib.pyplot as plt

# ------------------------- CONFIG ------------------------- #
CSV_PATH = "misc_analysis/Novartis_PDX_Control.csv"   # path to your CSV
TUMOUR_TYPE = "CRC"                # choose one tumour type
TREATMENT = "untreated"

# ------------------------- LOAD DATA ---------------------- #
df = pd.read_csv(CSV_PATH)

df_ctrl = df[
    (df["Treatment"] == TREATMENT) &
    (df["TumourType"] == TUMOUR_TYPE)
].copy()

print(f"Using tumour type: {TUMOUR_TYPE}")
print(f"Number of untreated controls: {len(df_ctrl)}")

# ------------------------- PLOT 1 ------------------------- #
# Histogram: Time to Double
plt.figure(figsize=(6.5, 4.5))
plt.hist(
    df_ctrl["TimeToDouble"].dropna(),
    bins=15,
    edgecolor="black",
    color="grey",
    alpha=0.7
)

plt.xlabel("Time to double (days)")
plt.ylabel("Number of tumours")
plt.title("Growth heterogeneity in untreated CRC tumours")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("misc_analysis/graphs/Novartis_TimeToDouble_Histogram.png", dpi=300)
plt.show()

# ------------------------- PLOT 2 ------------------------- #
# Scatter: Time to Double vs Best Response
plt.figure(figsize=(6.5, 4.5))
plt.scatter(
    df_ctrl["TimeToDouble"],
    df_ctrl["BestResponse"],
    alpha=0.75
)

plt.xlabel("Time to double (days)")
plt.ylabel("Best response (%)")
plt.title("Inter-tumour variability in untreated CRC controls")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("misc_analysis/graphs/Novartis_TimeToDouble_vs_BestResponse_Scatter.png", dpi=300)
plt.show()



plt.figure(figsize=(7,4))

df_untreated = df[df["Treatment"] == "untreated"]

df_untreated.boxplot(
    column="TimeToDouble",
    by="TumourType",
    grid=False
)

plt.ylabel("Time to double (days)")
plt.title("Growth heterogeneity across tumour types (untreated)")
plt.suptitle("")
plt.tight_layout()
plt.savefig("misc_analysis/graphs/Novartis_TumourType_Boxplot.png", dpi=300)
plt.show()