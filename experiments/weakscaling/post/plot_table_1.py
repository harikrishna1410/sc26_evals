#!/usr/bin/env python3
"""
Plot stacked vertical bar chart showing task latency breakdown at 1024 nodes across different sleep times.
"""

from glob import glob

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from plot_config import get_full_column_fig, get_half_column_fig
from utils import compute_completion_latency, compute_scheduling_latency

FORMAT = "pdf"


if __name__ == "__main__":
    node = 128
    frameworks = [
        "dask_1level_with_trigger",
        # "parsl_with_trigger",
        "dask_2level_processpool_with_trigger",
    ]
    label = {
        "dask_1level_with_trigger": "Depth=0",
        "dask_2level_processpool_with_trigger": "Depth=1",
        "parsl_with_trigger": "Parsl",
    }
    nworkers = 10
    sleep_times = [100, 60000]  # in milliseconds

    # Hatches for different frameworks
    hatches = [
        "\\\\",
        "ooo",
        "///",
    ]

    # Create figure
    fig, ax = get_half_column_fig(figsize=(1.75, 1.5))

    # Data structure: {sleep_time: {framework: {'scheduling': ..., 'elapsed': ..., 'completion': ...}}}
    data = {}

    for sleep_time in sleep_times:
        data[sleep_time] = {}

        for f in frameworks:
            dirname_fmt = f"../{f}/{node}/all_logs/logs_64_{nworkers}_{sleep_time}ms_*"
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
                    print(f"Warning: Could not process {dirname} for elapsed time: {e}")
                    continue

            # Store means and stds
            data[sleep_time][f] = {
                "scheduling": np.mean(scheduling_latencies)
                if scheduling_latencies
                else 0,
                "scheduling_std": np.std(scheduling_latencies)
                if scheduling_latencies
                else 0,
                "elapsed": np.mean(elapsed_times) if elapsed_times else 0,
                "elapsed_std": np.std(elapsed_times) if elapsed_times else 0,
                "completion": np.mean(completion_latencies)
                if completion_latencies
                else 0,
                "completion_std": np.std(completion_latencies)
                if completion_latencies
                else 0,
            }

    # Write plotted data (mean and std) to CSV
    csv_rows = []
    for st in sleep_times:
        for f in frameworks:
            scheduling_mean = data[st][f]["scheduling"]
            scheduling_std = data[st][f]["scheduling_std"]
            # Match the plotted quantity: mean(scheduling)/4.5 - sleep_time
            scheduling_overhead_mean = scheduling_mean / 4.5 - (st / 1000.0)
            # sleep_time is a constant offset; std of overhead = std(scheduling)/4.5
            scheduling_overhead_std = scheduling_std / 4.5

            csv_rows.append(
                {
                    "sleep_time_ms": st,
                    "framework": f,
                    "label": label[f],
                    "scheduling_overhead_mean": scheduling_overhead_mean,
                    "scheduling_overhead_std": scheduling_overhead_std,
                    "elapsed_mean": data[st][f]["elapsed"],
                    "elapsed_std": data[st][f]["elapsed_std"],
                    "completion_mean": data[st][f]["completion"],
                    "completion_std": data[st][f]["completion_std"],
                }
            )
    
    os.makedirs("data",exist_ok=True)
    csv_path = "data/table_1.csv"
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    print(f"Data saved to {csv_path}")

    # # Set up bar positions
    # n_sleep_times = len(sleep_times)
    # n_frameworks = len(frameworks)
    # bar_width = 0.25
    # x = np.arange(n_sleep_times)

    # # Colors for the three components
    # colors = {
    #     "scheduling_overhead": "#1f77b4",
    #     "elapsed": "#ff7f0e",
    #     "completion": "#2ca02c",
    # }

    # # Plot stacked bars for each framework
    # for i, (f, hatch) in enumerate(zip(frameworks, hatches)):
    #     positions = x + (i - n_frameworks / 2 + 0.5) * bar_width

    #     scheduling_overhead_vals = []
    #     elapsed_vals = []
    #     completion_vals = []

    #     for st in sleep_times:
    #         # Get the actual measurements
    #         scheduling = data[st][f]["scheduling"]
    #         elapsed = data[st][f]["elapsed"]
    #         completion = data[st][f]["completion"]

    #         # Calculate scheduling overhead: mean(scheduling_latency)/4.5 - sleep_time
    #         scheduling_overhead = scheduling / 4.5 - (st / 1000.0)

    #         scheduling_overhead_vals.append(scheduling_overhead)
    #         elapsed_vals.append(elapsed)
    #         completion_vals.append(completion)

    #     # Plot stacked bars
    #     p1 = ax.bar(
    #         positions,
    #         scheduling_overhead_vals,
    #         bar_width,
    #         label=label[f] if i == 0 else "",
    #         color=colors["scheduling_overhead"],
    #         hatch=hatch,
    #         edgecolor="black",
    #         linewidth=0.5,
    #     )
    #     p2 = ax.bar(
    #         positions,
    #         elapsed_vals,
    #         bar_width,
    #         bottom=scheduling_overhead_vals,
    #         color=colors["elapsed"],
    #         hatch=hatch,
    #         edgecolor="black",
    #         linewidth=0.5,
    #     )
    #     p3 = ax.bar(
    #         positions,
    #         completion_vals,
    #         bar_width,
    #         bottom=np.array(scheduling_overhead_vals) + np.array(elapsed_vals),
    #         color=colors["completion"],
    #         hatch=hatch,
    #         edgecolor="black",
    #         linewidth=0.5,
    #     )

    # # Axes legend: component colors
    # component_elements = [
    #     Patch(facecolor=colors["scheduling_overhead"], label="Scheculing Overhead"),
    #     Patch(facecolor=colors["elapsed"], label="Execution Time"),
    #     Patch(facecolor=colors["completion"], label="Completion Detection"),
    # ]
    # # ax.legend(handles=component_elements, loc="upper left")

    # # Figure legend: framework hatches
    # hatch_elements = [
    #     Patch(facecolor="white", edgecolor="black", hatch=hatch, label=label[f])
    #     for f, hatch in zip(frameworks, hatches)
    # ]
    # fig.legend(
    #     handles=hatch_elements + component_elements,
    #     loc="center left",
    #     ncol=1,
    #     handlelength=1.0,
    #     bbox_to_anchor=(1.0, 0.5),
    # )

    # # Configure plot
    # ax.set_xticks(x)
    # ax.set_xticklabels([f"{st / 1000.0}" for st in sleep_times])
    # ax.set_xlabel("Duration (s)")
    # ax.set_ylabel("Time (s)")
    # ax.set_title(f"{node} Nodes")
    # ax.grid(True, alpha=0.3, axis="y")
    # # ax.set_yscale("log")

    # plt.tight_layout()
    # fig.savefig(
    #     f"figs/motivation/task_latency_breakdown_{node}nodes.{FORMAT}",
    #     dpi=300,
    #     bbox_inches="tight",
    # )
    # print(f"Plot saved to figs/motivation/task_latency_breakdown_{node}nodes.{FORMAT}")
    # plt.show()

    # print(
    #     f"Plot saved to figs/motivation/task_latency_breakdown_{node}nodes_{sleep_time}ms.{FORMAT}"
    # )
    # plt.show()
