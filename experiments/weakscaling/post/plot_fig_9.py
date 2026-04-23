"""
Resource Utilisation Ablation Plot

Plots CPU utilisation for the same frameworks as plot_scaling_ablation.py:
Dask(Depth=1), EL(Cluster), EL(Cluster, Depth=1), plus work-stealing ablation variants.
"""

import argparse
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from plot_config import (
    FRAMEWORK_COLORS,
    FRAMEWORK_MARKERS,
    LABELS_DIRNAMES,
    get_three_quarter_fig,
)
from utils import determine_system_type

FORMAT = "pdf"


def compute_utilisation(timeline_csv):
    df = pd.read_csv(timeline_csv)

    start_times = df["start_time(s)"].values
    end_times = df["end_time(s)"].values

    times = np.concatenate([start_times, end_times])
    changes = np.concatenate([np.ones(len(start_times)), -np.ones(len(end_times))])

    idx = np.argsort(times)
    times = times[idx]
    utilisation = np.cumsum(changes[idx])

    # Shift to start at a small positive value (for log scale compatibility)
    times = times - times[0]
    times[times <= 0] = 1e-4
    return times, utilisation


def plot_utilisation(
    dirnames,
    labels,
    node_count,
    nworkers=102,
    ntasks_per_worker=10,
    sleep_time=1000,
    trial=1,
    el_level=2,
):
    fig, ax = get_three_quarter_fig(figsize=(2.5, 2.0))
    max_cpus = nworkers * node_count

    for base_dir, label in zip(dirnames, labels):
        system_type = determine_system_type(base_dir)
        if system_type == "el" or system_type == "el_cluster":
            lvl = 1 if "1level" in base_dir else el_level
            log_dir = f"{base_dir}/{node_count}/all_logs/logs_{nworkers}_{ntasks_per_worker}_{sleep_time}ms_{lvl}levels_{trial}"
        elif system_type == "flux":
            log_dir = f"{base_dir}/{node_count}/all_logs/logs_{nworkers}_{ntasks_per_worker}_{sleep_time}ms_{trial}"
        else:
            log_dir = f"{base_dir}/{node_count}/all_logs/logs_{nworkers}_{ntasks_per_worker}_{sleep_time}ms_{trial}"

        timeline_csv = os.path.join(log_dir, "timeline_worker.csv")
        if not os.path.isfile(timeline_csv):
            print(f"Skipping {label}: {timeline_csv} not found")
            continue

        times, utilisation = compute_utilisation(timeline_csv)
        color = FRAMEWORK_COLORS.get(label, None)
        if len(times) < 2 * node_count * nworkers * ntasks_per_worker:
            continue

        utilisation_pct = utilisation / max_cpus * 100
        ax.step(
            times[-2 * node_count * nworkers * ntasks_per_worker :]
            - times[-2 * node_count * nworkers * ntasks_per_worker],
            utilisation_pct[-2 * node_count * nworkers * ntasks_per_worker :],
            where="post",
            color=color,
            label=label,
            linewidth=1,
        )
        print(f"Done plotting {base_dir}, {len(times)}, {timeline_csv}")

    return fig, ax


def plot_ablation_utilisation(
    fig,
    ax,
    node_count,
    sleep_time,
    color,
    level=1,
    batch_sizes=[0, 1020],
    nworkers=102,
    ntasks_per_worker=10,
    trial=1,
):
    max_cpus = nworkers * node_count
    base_dir = f"../el_cluster_{level}level_ws"
    ln_st = [":", "-.", "--"]
    for idx, batch_size in enumerate(batch_sizes):
        log_dir = f"{base_dir}/{node_count}/all_logs/logs_{nworkers}_{ntasks_per_worker}_{sleep_time}ms_{level}levels_{batch_size}_{trial}"
        timeline_csv = os.path.join(log_dir, "timeline_worker.csv")
        if not os.path.isfile(timeline_csv):
            print(
                f"Skipping level={level}, batch={batch_size}: {timeline_csv} not found"
            )
            continue

        times, utilisation = compute_utilisation(timeline_csv)
        utilisation_pct = utilisation / max_cpus * 100
        ax.step(
            times,
            utilisation_pct,
            where="post",
            color=color,
            linestyle=ln_st[idx],
            label=f"EL(level={level}, Batch={batch_size})",
            linewidth=1,
        )

    return fig, ax


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--node-count", type=int, default=128, help="Number of nodes")
    parser.add_argument("--nworkers", type=int, default=102)
    parser.add_argument("--ntasks-per-worker", type=int, default=10)
    parser.add_argument(
        "--sleep-time", type=int, default=1000, help="Sleep time in ms (default: 1000)"
    )
    parser.add_argument("--trial", type=int, default=1)
    args = parser.parse_args()

    dirnames = []  # ["../dask_2level_processpool_with_trigger"]
    labels = [LABELS_DIRNAMES[d] for d in dirnames]
    print(labels)

    fig, ax = plot_utilisation(
        dirnames,
        labels,
        node_count=args.node_count,
        nworkers=args.nworkers,
        ntasks_per_worker=args.ntasks_per_worker,
        sleep_time=args.sleep_time,
        trial=args.trial,
    )

    fig, ax = plot_ablation_utilisation(
        fig,
        ax,
        args.node_count,
        args.sleep_time,
        FRAMEWORK_COLORS["EL(Cluster, Depth = 1)"],
        level=1,
        batch_sizes=[0],
        nworkers=args.nworkers,
        ntasks_per_worker=args.ntasks_per_worker,
        trial=args.trial,
    )
    fig, ax = plot_ablation_utilisation(
        fig,
        ax,
        args.node_count,
        args.sleep_time,
        FRAMEWORK_COLORS["EL(Cluster)"],
        level=2,
        batch_sizes=[0],
        nworkers=args.nworkers,
        ntasks_per_worker=args.ntasks_per_worker,
        trial=args.trial,
    )

    ax.axhline(
        100,
        color="black",
        linestyle=":",
        linewidth=0.75,
        label="Max (100%)",
    )
    ax.set_xscale("log", base=10)
    ax.set_yscale("log", base=10)
    ax.set_xlim(left=0.1)
    ax.set_ylim(bottom=0.1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("CPU Utilisation (%)")
    ax.set_title(
        f"Nodes = {args.node_count}, Duration (s) = {args.sleep_time / 1000:.1f}"
    )
    ax.grid(True, alpha=0.3)

    # Axes legend: frameworks
    ax_fw_handles = [
        # Line2D(
        #     [0],
        #     [0],
        #     color=FRAMEWORK_COLORS["Dask(Depth=1)"],
        #     linestyle="-",
        #     linewidth=1,
        #     label="Dask(Depth=1)",
        # ),
        Line2D(
            [0],
            [0],
            color=FRAMEWORK_COLORS["EL(Cluster)"],
            linestyle="-",
            linewidth=1,
            label="EL(Depth=2)",
        ),
        Line2D(
            [0],
            [0],
            color=FRAMEWORK_COLORS["EL(Cluster, Depth = 1)"],
            linestyle="-",
            linewidth=1,
            label="EL(Depth=1)",
        ),
    ]
    # ax.legend(handles=ax_fw_handles, loc="best", fontsize=10)

    # Figure legend: push/pull variants
    # batch_handles = [
    #     Line2D([0], [0], color="black", linestyle="-", linewidth=1, label="Push"),
    #     Line2D([0], [0], color="black", linestyle=":", linewidth=1, label="Pull"),
    # ]
    fig.legend(
        handles=ax_fw_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=2,
        handlelength=1.5,
        fontsize=10,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.89])
    # fig.tight_layout()

    # os.makedirs("figs/ablation", exist_ok=True)
    fname = "figs/fig_9"
    fig.savefig(f"{fname}.{FORMAT}", dpi=300)
    print(f"Saved: {fname}.{FORMAT}")
    plt.close(fig)
