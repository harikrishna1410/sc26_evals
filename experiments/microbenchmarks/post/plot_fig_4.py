"""
Throughput vs Number of Tasks Plot
Four half-column subplots in a single figure, one per sleep time.
Reads all CSV files from data/ directory directly.
"""

import glob
import os

import pandas as pd
import scienceplots
from matplotlib import pyplot as plt

from plot_config_wt import (
    FRAMEWORK_COLORS,
    FRAMEWORK_LINESTYLES,
    FRAMEWORK_MARKERS,
    get_half_column_fig,
)

FORMAT = "pdf"

LABEL_MAP = {
    "": "Processpool",
    "dask": "Dask",
    "parsl-htex": "Parsl",
    "worker": "EL(1)",
    "worker-master": "EL(102)",
}

# Load all CSVs from data/
csv_files = sorted(glob.glob("../worker_throughput/data/*.csv"))
datasets = {}
for path in csv_files:
    name = (
        os.path.basename(path)
        .replace("throughput_results_v2", "")
        .strip("_")
        .replace(".csv", "")
    )
    key = name.replace("_", "-") if name else ""
    if key not in LABEL_MAP:
        continue
    label = LABEL_MAP[key]
    df = pd.read_csv(path)
    # Aggregate across repeat runs so we can plot mean ± std error bars.
    df = (
        df.groupby(["sleeptime", "ntasks"], as_index=False)["throughput"]
        .agg(throughput_mean="mean", throughput_std="std")
        .sort_values(["sleeptime", "ntasks"])
    )
    df["throughput_std"] = df["throughput_std"].fillna(0.0)
    datasets[label] = df

sleeptimes = [0.0, 1.0]

fig, axes = get_half_column_fig(ncols=2, figsize=(3.5, 1.5))
axes = axes.flatten()

for ax_idx, sleeptime in enumerate(sleeptimes):
    ax = axes[ax_idx]
    for label, df in datasets.items():
        subset = df[df["sleeptime"] == sleeptime].sort_values("ntasks")
        if subset.empty:
            continue
        ax.errorbar(
            subset["ntasks"],
            subset["throughput_mean"],
            yerr=subset["throughput_std"],
            marker=FRAMEWORK_MARKERS.get(label, "o"),
            color=FRAMEWORK_COLORS.get(label, "black"),
            label=label,
            linestyle=FRAMEWORK_LINESTYLES.get(label, "-"),
            capsize=2,
            elinewidth=0.8,
        )
    # if ax_idx >= 2:
    ax.set_xlabel("Number of tasks")
    if ax_idx % 2 == 0:
        ax.set_ylabel("Throughput (tasks/s)")
    ax.set_title(f"Duration (s) = {sleeptime}")
    ax.set_xscale("log", base=10)
    ax.set_yscale("log", base=10)
    ax.xaxis.set_major_locator(plt.matplotlib.ticker.FixedLocator([10, 100, 1000, 10000, 100000]))
    ax.xaxis.set_major_formatter(plt.matplotlib.ticker.LogFormatterSciNotation(base=10))

# Single shared legend
handles, labels = axes[0].get_legend_handles_labels()
legend = fig.legend(
    handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.21), fontsize=10
)
for line in legend.get_lines():
    line.set_linewidth(1.5)

fig.tight_layout()

output_file = f"fig_4.{FORMAT}"
fig.savefig(output_file, dpi=300, bbox_inches="tight")
print(f"Saved: {output_file}")
