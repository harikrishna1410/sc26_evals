"""
Scaling Ablation Analysis Plotting Script

This script reads log files from different task execution frameworks
(EL, MPI, Parsl, Dask) and generates scaling performance plots
with batch size ablation.
"""

import os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter

from plot_config import (
    FRAMEWORK_COLORS,
    FRAMEWORK_MARKERS,
    LABELS_DIRNAMES,
    get_three_quarter_fig,
)
from utils import (
    determine_system_type,
    read_el_cluster_log,
    read_el_log,
    read_flux_log,
    read_framework_log,
    read_mpi_log,
    save_plot_data_to_csv,
)

FORMAT = "pdf"

LOG_READERS = {
    "dask": read_framework_log,
    "parsl": read_framework_log,
    "mpi": read_mpi_log,
    "flux": read_flux_log,
    "el": read_el_log,
    "el_cluster": read_el_cluster_log,
}


def _get_log_path(
    system_type,
    base_dir,
    nn,
    nworkers,
    ntasks_per_worker,
    sleep_time,
    trial,
    el_level=2,
):
    if system_type == "mpi":
        return f"{base_dir}/{nn}/timings/elapsed_time_{nworkers}_{ntasks_per_worker}_{sleep_time}ms.txt"
    elif system_type == "el" or system_type == "el_cluster":
        return f"{base_dir}/{nn}/all_logs/logs_{nworkers}_{ntasks_per_worker}_{sleep_time}ms_{el_level}levels_{trial}"
    elif system_type == "flux":
        return f"{base_dir}/{nn}/all_logs/logs_{nworkers}_{ntasks_per_worker}_{sleep_time}ms_{trial}"
    else:
        return f"{base_dir}/{nn}/all_logs/logs_{nworkers}_{ntasks_per_worker}_{sleep_time}ms_{trial}/main.log"


def plot_elapsed_time(
    nnodes,
    sleep_time=1,
    nworkers=12,
    dirnames=["el_scaling_zmq", "parsl_scaling"],
    labels=["EL", "PARSL"],
    ntasks_per_worker=1,
):
    """
    Plot elapsed time vs. number of nodes with ideal line.

    Returns:
        tuple: (figure, axes, plot_data)
    """
    fig, ax = get_three_quarter_fig()

    plot_data = {}

    for base_dir, label in zip(dirnames, labels):
        system_type = determine_system_type(base_dir)
        log_reader = LOG_READERS[system_type]
        el_level = 2 if system_type == "el" or system_type == "el_cluster" else None
        if "1level" in base_dir:
            el_level = 1

        means, stds, nodes_plt = [], [], []

        ntrials = 1 if system_type == "mpi" else 3
        for nn in nnodes:
            times = []
            for trial in range(1, ntrials + 1):
                fname = _get_log_path(
                    system_type,
                    base_dir,
                    nn,
                    nworkers,
                    ntasks_per_worker,
                    sleep_time,
                    trial,
                    el_level,
                )
                try:
                    times.extend(log_reader(fname))
                except FileNotFoundError:
                    continue
                except Exception as e:
                    print(f"Error reading {fname}: {e}")

            if times:
                means.append(np.mean(times))
                stds.append(np.std(times))
                nodes_plt.append(nn)

        marker = FRAMEWORK_MARKERS.get(label, "o")
        ax.errorbar(
            nodes_plt,
            means,
            yerr=stds,
            fmt=f"{marker}-",
            label=f"{label}",
            color=FRAMEWORK_COLORS[label],
        )
        plot_data[label] = {"nodes": nodes_plt, "mean": means, "std": stds}

    ideal_time = sleep_time * ntasks_per_worker / 1000  # ms -> s
    if sleep_time > 0.0:
        ax.axhline(
            y=ideal_time, color="black", linestyle=":", label="Ideal"
        )

    ax.set_xlabel("Number of nodes")
    ax.set_ylabel("Elapsed time (seconds)")
    ax.set_xscale("log", base=2)
    ax.set_title(f"Duration (s) = {sleep_time / 1000:.1f}")
    if sleep_time < 600000:
        ax.set_yscale("log", base=10)
    ax.set_ylim(bottom=10)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xticks(nnodes)
    ax.tick_params(axis="x", labelrotation=90)
    ax.grid(True, alpha=0.3)
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    return fig, ax, plot_data


def plot_ablation_batch_size(fig, ax, nnodes, sleeptime, color, level=1, batch_sizes=[0, 510, 1020]):
    """
    Plot elapsed time vs. number of nodes for batch size ablation.

    Returns:
        tuple: (figure, axes)
    """
    base_dir = f"../el_cluster_{level}level_ws"
    log_reader = LOG_READERS["el_cluster"]
    ln_st = [":", "-.", "--"]
    for idx, batch_size in enumerate(batch_sizes):
        means, stds, nodes_plt = [], [], []
        for nn in nnodes:
            times = []
            for trial in range(1, 4):
                try:
                    fname = base_dir + f"/{nn}/all_logs/logs_102_10_{sleeptime}ms_{level}levels_{batch_size}_{trial}"
                    times.extend(log_reader(fname))
                    print(f"Read: {fname}")
                except Exception:
                    pass

            if times:
                means.append(np.mean(times))
                stds.append(np.std(times))
                nodes_plt.append(nn)

        label = f"EL(level={level}, Batch={batch_size})"
        ax.errorbar(
            nodes_plt,
            means,
            yerr=stds,
            fmt=f"o{ln_st[idx]}",
            label=f"{label}",
            color=color,
        )

    return fig, ax


if __name__ == "__main__":
    nnodes = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    dirnames = [
        "../dask_2level_processpool_with_trigger",
        # "../el_cluster",
        # "../el_cluster_1level",
    ]

    os.makedirs("figs", exist_ok=True)

    for nworkers in [102]:
        for sleep_time in [1000]:
            for ntasks_per_worker in [10]:
                ext = f"{nworkers}_{ntasks_per_worker}_{sleep_time}ms"

                print(f"Processing {ext}: parsing log files...")
                fig, ax, plot_data = plot_elapsed_time(
                    nnodes,
                    sleep_time=sleep_time,
                    nworkers=nworkers,
                    dirnames=dirnames,
                    labels=[LABELS_DIRNAMES[d] for d in dirnames],
                    ntasks_per_worker=ntasks_per_worker,
                )
                fig, ax = plot_ablation_batch_size(fig, ax, nnodes, sleep_time, FRAMEWORK_COLORS["EL(Cluster, Depth = 1)"], level=1, batch_sizes=[0])
                fig, ax = plot_ablation_batch_size(fig, ax, nnodes, sleep_time, FRAMEWORK_COLORS["EL(Cluster)"], level=2, batch_sizes=[0, 1020])

                # Axes legend: frameworks (colored solid handles)
                ax_fw_handles = [
                    Line2D([0], [0], color=FRAMEWORK_COLORS["Dask(Depth=1)"], marker=FRAMEWORK_MARKERS["Dask(Depth=1)"], linestyle='-', markersize=2, linewidth=1, label="Dask(Depth=1)"),
                    Line2D([0], [0], color=FRAMEWORK_COLORS["EL(Cluster)"], marker=FRAMEWORK_MARKERS["EL(Cluster)"], linestyle='-', markersize=2, linewidth=1, label="EL(Depth=2)"),
                    Line2D([0], [0], color=FRAMEWORK_COLORS["EL(Cluster, Depth = 1)"], marker=FRAMEWORK_MARKERS["EL(Cluster, Depth = 1)"], linestyle='-', markersize=2, linewidth=1, label="EL(Depth=1)"),
                ]
                ax.legend(handles=ax_fw_handles, loc="best", fontsize=10, handlelength=1.0)

                # Figure legend: batch sizes (black handles, different line styles)
                batch_handles = [
                    # Line2D([0], [0], color='black', linestyle='-', linewidth=1, label="Push"),
                    Line2D([0], [0], color='black', linestyle=':', linewidth=1, label="Pull"),
                    Line2D([0], [0], color='black', linestyle='-.', linewidth=1, label="Pull(Batch=1020)"),
                ]
                fig.legend(handles=batch_handles, loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=3, handlelength=1.5, fontsize=10)

                os.makedirs("figs", exist_ok=True)
                plot_file = f"figs/fig_8.{FORMAT}"
                fig.savefig(plot_file, dpi=300)
                print(f"Saved plot: {plot_file}")

                plt.close(fig)
