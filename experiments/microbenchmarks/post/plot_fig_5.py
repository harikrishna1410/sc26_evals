"""
CDF of task roundtrip latency.
One half-column figure per framework, all on a single plot.
"""

import glob
import os

import numpy as np
import pandas as pd
import scienceplots
from matplotlib import pyplot as plt

from plot_config_rl import FRAMEWORK_COLORS, FRAMEWORK_MARKERS, get_three_quarter_fig

FORMAT = "pdf"

LABEL_MAP = {
    "processpool": "Processpool",
    "dask": "Dask",
    "parsl": "Parsl",
    "worker": "EL(Launcher)",
    "ensemble_master": "EL(Manager)",
}

# Load all task CSVs
csv_files = sorted(glob.glob("../roundtrip_latency/data/*_tasks.csv"))
datasets = {}
for path in csv_files:
    # Extract framework name from filename
    basename = os.path.basename(path)
    # Pattern: latency_results_10000_sequential_<framework>_tasks.csv
    name = basename.replace("latency_results_10000_sequential_", "").replace(
        "_tasks.csv", ""
    )
    if name not in LABEL_MAP:
        continue
    label = LABEL_MAP[name]
    df = pd.read_csv(path)
    datasets[label] = df

FRAMEWORK_LINESTYLES = {
    "EL(Launcher)": "-",
    "EL(Manager)": "--",
}

fig, ax = get_three_quarter_fig(figsize=(2.5, 1.5))

for label, df in datasets.items():
    latencies = np.sort(df["elapsed_time(s)"].values) * 1000  # convert to ms
    cdf = np.arange(1, len(latencies) + 1) / len(latencies)
    ax.plot(
        latencies,
        cdf,
        color=FRAMEWORK_COLORS.get(label, "black"),
        linestyle=FRAMEWORK_LINESTYLES.get(label, "-"),
        label=label,
    )

ax.set_xlabel("Roundtrip latency (ms)")
ax.set_ylabel("CDF")
ax.set_xscale("log", base=10)
ax.legend(loc="best", fontsize=8)
fig.tight_layout()

output_file = f"fig_5.{FORMAT}"
fig.savefig(output_file, dpi=300, bbox_inches="tight")
print(f"Saved: {output_file}")
