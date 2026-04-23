#!/usr/bin/env python3
"""
Plot stacked vertical bar chart showing task latency breakdown at 1024 nodes across different sleep times.
"""

from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from plot_config import get_full_column_fig, get_half_column_fig, get_three_quarter_fig
from utils import compute_completion_latency, compute_scheduling_latency

FORMAT = "pdf"


if __name__ == "__main__":
    nodes = [128, 1024]
    frameworks = [
        # "dask_1level_with_trigger",
        "parsl_with_trigger",
        "dask_2level_processpool_with_trigger",
    ]
    label = {
        # "dask_1level_with_trigger": "Dask(Depth=0)",
        "dask_2level_processpool_with_trigger": "Dask(Depth=1)",
        "parsl_with_trigger": "Parsl",
    }
    nworkers = 10
    sleep_times = [100, 1000]  # in milliseconds

    # Hatches for different frameworks
    hatches = [
        "\\\\",
        "ooo",
        "///",
    ]

    # Colors for the three components
    colors = {
        "scheduling_overhead": "#1f77b4",
        "elapsed": "#ff7f0e",
        "completion": "#2ca02c",
    }

    # Create figure with two subplots
    fig, axes = get_full_column_fig(ncols=len(nodes), figsize=(3.5, 2))

    for ax_idx, node in enumerate(nodes):
        ax = axes[ax_idx]

        # Data structure: {sleep_time: {framework: {'scheduling': ..., 'elapsed': ..., 'completion': ...}}}
        data = {}

        for sleep_time in sleep_times:
            data[sleep_time] = {}

            for f in frameworks:
                dirname_fmt = (
                    f"../{f}/{node}/all_logs/logs_102_{nworkers}_{sleep_time}ms_*"
                )
                dirnames = glob(dirname_fmt)

                # Collect scheduling latencies
                scheduling_latencies = []
                for dirname in dirnames:
                    try:
                        scheduling_latencies.extend(compute_scheduling_latency(dirname))
                    except Exception as e:
                        print(
                            f"Warning: Could not process {dirname} for scheduling latency: {e}"
                        )
                        continue

                # Collect completion latencies
                completion_latencies = []
                for dirname in dirnames:
                    try:
                        completion_latencies.extend(compute_completion_latency(dirname))
                    except Exception as e:
                        print(
                            f"Warning: Could not process {dirname} for completion latency: {e}"
                        )
                        continue

                # Read elapsed time from timeline_worker.csv
                elapsed_times = []
                for dirname in dirnames:
                    try:
                        worker_timeline = f"{dirname}/timeline_worker.csv"
                        df = pd.read_csv(worker_timeline)
                        if "elapsed_time(s)" in df.columns:
                            elapsed_times.extend(df["elapsed_time(s)"].tolist())
                    except Exception as e:
                        print(
                            f"Warning: Could not process {dirname} for elapsed time: {e}"
                        )
                        continue

                # Store means
                data[sleep_time][f] = {
                    "scheduling": np.mean(scheduling_latencies)
                    if scheduling_latencies
                    else 0,
                    "elapsed": np.mean(elapsed_times) if elapsed_times else 0,
                    "completion": np.mean(completion_latencies)
                    if completion_latencies
                    else 0,
                }

        # Set up bar positions
        n_sleep_times = len(sleep_times)
        n_frameworks = len(frameworks)
        bar_width = 0.25
        x = np.arange(n_sleep_times)

        # Plot stacked bars for each framework
        for i, (f, hatch) in enumerate(zip(frameworks, hatches)):
            positions = x + (i - n_frameworks / 2 + 0.5) * bar_width

            scheduling_overhead_vals = []
            elapsed_vals = []
            completion_vals = []

            for st in sleep_times:
                # Get the actual measurements
                scheduling = data[st][f]["scheduling"]
                elapsed = data[st][f]["elapsed"]
                completion = data[st][f]["completion"]

                # Calculate scheduling overhead: mean(scheduling_latency)/4.5 - sleep_time
                scheduling_overhead = scheduling / 4.5 - (st / 1000.0)

                scheduling_overhead_vals.append(scheduling_overhead)
                elapsed_vals.append(elapsed)
                completion_vals.append(completion)

            # Plot stacked bars
            ax.bar(
                positions,
                scheduling_overhead_vals,
                bar_width,
                color=colors["scheduling_overhead"],
                hatch=hatch,
                edgecolor="black",
                linewidth=0.5,
            )
            ax.bar(
                positions,
                elapsed_vals,
                bar_width,
                bottom=scheduling_overhead_vals,
                color=colors["elapsed"],
                hatch=hatch,
                edgecolor="black",
                linewidth=0.5,
            )
            ax.bar(
                positions,
                completion_vals,
                bar_width,
                bottom=np.array(scheduling_overhead_vals) + np.array(elapsed_vals),
                color=colors["completion"],
                hatch=hatch,
                edgecolor="black",
                linewidth=0.5,
            )

            # Annotate overhead percentage on top of each bar
            for j in range(len(positions)):
                total = (
                    scheduling_overhead_vals[j] + elapsed_vals[j] + completion_vals[j]
                )
                overhead = scheduling_overhead_vals[j] + completion_vals[j]
                if total > 0:
                    pct = overhead / total * 100
                    ax.text(
                        positions[j],
                        total,
                        f"{pct:.1f}%",
                        ha="center",
                        va="bottom",
                        fontsize=6,
                    )

        # Configure subplot
        ax.set_xticks(x)
        ax.set_xticklabels([f"{st / 1000.0}" for st in sleep_times])
        ax.set_xlabel("Duration (s)")
        if ax_idx == 0:
            ax.set_ylabel("Time (s)")
        ax.set_title(f"{node} Nodes")
        ax.grid(True, alpha=0.3, axis="y")

    # Figure legend: component colors + framework hatches
    component_elements = [
        Patch(facecolor=colors["scheduling_overhead"], label="SO"),
        Patch(facecolor=colors["elapsed"], label="ET"),
        Patch(facecolor=colors["completion"], label="CD"),
    ]
    hatch_elements = [
        Patch(facecolor="white", edgecolor="black", hatch=hatch, label=label[f])
        for f, hatch in zip(frameworks, hatches)
    ]
    all_legend_elements = component_elements + hatch_elements
    fig.legend(
        handles=all_legend_elements,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 1.15),
    )

    plt.tight_layout()
    fig.savefig(
        f"figs/fig_7.{FORMAT}",
        dpi=300,
        bbox_inches="tight",
    )
    print(
        f"Plot saved to figs/motivation/task_latency_breakdown_128_1024nodes.{FORMAT}"
    )
    plt.show()
