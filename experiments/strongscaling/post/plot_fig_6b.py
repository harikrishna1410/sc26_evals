"""
Scaling Analysis Plotting Script

This script reads log files from different task execution frameworks
(EL, MPI, Parsl, Dask) and generates scaling performance plots.
"""

import os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter

import plot_config  # noqa: F401
from plot_config import (
    FRAMEWORK_COLORS,
    FRAMEWORK_LINESTYLES,
    FRAMEWORK_MARKERS,
    LABELS_DIRNAMES,
    get_full_column_fig,
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
    "el_cluster_pull": read_el_cluster_log,
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
    elif system_type == "el_cluster_pull":
        return f"{base_dir}/{nn}/all_logs/logs_{nworkers}_{ntasks_per_worker}_{sleep_time}ms_{el_level}levels_0_{trial}"
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
    key_colors=None,
    ntasks_per_worker=1,
    ax=None,
):
    """
    Plot elapsed time vs. number of nodes with ideal line.

    Returns:
        tuple: (figure, axes, plot_data)
    """
    if ax is None:
        fig, ax = get_full_column_fig(figsize=(3.5,1.5))
    else:
        fig = ax.figure

    if key_colors is None:
        key_colors = FRAMEWORK_COLORS

    plot_data = {}

    for base_dir, label in zip(dirnames, labels):
        system_type = determine_system_type(base_dir)
        log_reader = LOG_READERS[system_type]
        el_level = 2 if system_type == "el" or system_type == "el_cluster" or system_type=="el_cluster_pull" else None
        if "1level" in base_dir:
            el_level = 1

        means, stds, nodes_plt = [], [], []

        for nn in nnodes:
            times = []
            try:
                if system_type == "mpi":
                    fname = _get_log_path(
                        system_type,
                        base_dir,
                        nn,
                        nworkers,
                        ntasks_per_worker,
                        sleep_time,
                        trial=None,
                    )
                    try:
                        times.extend(log_reader(fname))
                    except FileNotFoundError:
                        print(f"{fname} not found")
                        pass
                    except Exception as e:
                        print(f"Error reading {fname}: {e}")
                else:
                    for trial in range(1, 4):
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
                            print(f"{fname} not found")
                            continue
                        except Exception as e:
                            print(f"Error reading {fname}: {e}")
            except ValueError as e:
                print(f"Error reading: {e}")
                continue

            if times:
                means.append(np.mean(times))
                stds.append(np.std(times))
                nodes_plt.append(nn)

        marker = FRAMEWORK_MARKERS.get(label, "o")
        linestyle = FRAMEWORK_LINESTYLES.get(label, "-")
        ax.errorbar(
            nodes_plt,
            means,
            yerr=stds,
            marker=marker,
            linestyle=linestyle,
            label=f"{label}",
            color=key_colors[label],
        )
        plot_data[label] = {"nodes": nodes_plt, "mean": means, "std": stds}

    if sleep_time > 0.0:
        nodes_arr = np.array(nnodes)
        ideal = 2048 * (sleep_time / 1000.0) / nodes_arr
        ax.plot(nodes_arr, ideal, color="black", linestyle=":", label="Ideal")

    ax.set_xlabel("Number of nodes")
    ax.set_ylabel("Elapsed time (seconds)")
    ax.set_xscale("log", base=2)
    ax.set_title(f"Duration (s) = {sleep_time / 1000:.1f}")
    if sleep_time < 600000:
        ax.set_yscale("log", base=10)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xticks(nnodes)
    ax.tick_params(axis="x", labelrotation=90)
    ax.grid(True, alpha=0.3)

    return fig, ax, plot_data


if __name__ == "__main__":
    nnodes = [64, 128, 256, 512, 1024, 2048]
    dirnames = [
        "../el",
        "../el_cluster",
        "../el_cluster_2level_ws",
        # "../el_cluster_1level",
        "../mpi",
        "../parsl_with_trigger",
        "../dask_2level_processpool_with_trigger",
        # "../dask_1level_with_trigger",
    ]

    for nworkers in [102]:
        for ntasks_per_worker in [1]:
            sleep_times = [1000, 60000]
            fig, axes = get_full_column_fig(ncols=2,figsize=(3.5,2))
            for i, (ax, sleep_time) in enumerate(zip(axes, sleep_times)):
                ext = f"{nworkers}_{ntasks_per_worker}_{sleep_time}ms"
                csv_file = f"data/elapsed_time_{ext}_with_trigger.csv"

                print(f"Processing {ext}: parsing log files...")
                _, _, plot_data = plot_elapsed_time(
                    nnodes,
                    sleep_time=sleep_time,
                    nworkers=nworkers,
                    dirnames=dirnames,
                    labels=[LABELS_DIRNAMES[d] for d in dirnames],
                    ntasks_per_worker=ntasks_per_worker,
                    ax=ax,
                )
                if i != 0:
                    ax.set_ylabel("")
                # save_plot_data_to_csv(
                #     plot_data, csv_file, sleep_time, nworkers, ntasks_per_worker
                # )
                # print(f"Saved data: {csv_file}")

            fig.tight_layout(w_pad=0.3)

            ext = f"{nworkers}_{ntasks_per_worker}_combined"
            for fmt in (FORMAT,):
                plot_file = f"fig_6b.{fmt}"
                fig.savefig(plot_file, dpi=300, bbox_inches="tight")
                print(f"Saved plot: {plot_file}")
            plt.close(fig)
