import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plot_config
from plot_config import get_full_column_fig


def read_all_tasks(dirname: str, policy: str, trial: int):
    serial_csv = os.path.join(dirname, f"timeline_{policy}_{trial}.csv")
    mpi_csv = os.path.join(dirname, f"mpi_timeline_{policy}_{trial}.csv")

    frames = [pd.read_csv(serial_csv)]
    if os.path.exists(mpi_csv):
        frames.append(pd.read_csv(mpi_csv))
    return pd.concat(frames, ignore_index=True)


def get_global_start(dirname: str, policy: str, trial: int):
    df = read_all_tasks(dirname, policy, trial)
    return df["start_time(s)"].min()


def compute_utilisation(
    dirname: str, policy: str, trial: int, task_type: str = None, t0: float = None
):

    df = read_all_tasks(dirname, policy, trial)

    if task_type is not None:
        df = df[df["task_type"] == task_type]

    if len(df) == 0:
        return None, None, None

    start_times = df["start_time(s)"].values
    end_times = df["end_time(s)"].values
    cpus = df["ncpus"].values
    gpus = df["ngpus"].values

    times = np.concatenate([start_times, end_times])
    cpu_changes = np.concatenate([cpus, -cpus])
    gpu_changes = np.concatenate([gpus, -gpus])

    idx = np.argsort(times)
    times = times[idx]
    cpu_utilisation = np.cumsum(cpu_changes[idx])
    gpu_utilisation = np.cumsum(gpu_changes[idx])

    # Shift relative to global start
    if t0 is None:
        t0 = times[0]
    times = times - t0

    # Resample onto a uniform time grid and smooth with a rolling average
    t_uniform = np.linspace(times[0], times[-1], len(times))
    cpu_uniform = np.interp(t_uniform, times, cpu_utilisation)
    gpu_uniform = np.interp(t_uniform, times, gpu_utilisation)

    window = min(1000, len(t_uniform) // 5)
    if window > 1:
        kernel = np.ones(window) / window
        cpu_smooth = np.convolve(cpu_uniform, kernel, mode="same")
        gpu_smooth = np.convolve(gpu_uniform, kernel, mode="same")
    else:
        cpu_smooth = cpu_uniform
        gpu_smooth = gpu_uniform

    return t_uniform, cpu_smooth, gpu_smooth


def get_task_types(dirname: str, policy: str, trial: int):
    df = read_all_tasks(dirname, policy, trial)
    return sorted(df["task_type"].unique())


def plot_mofa_utilisation(
    policies=[
        "fixed_leafs_children_policy",
        "resource_split_policy",
        "routing_policy",
    ],
    labels=["Uniform", "Partitioned", "Adaptive"],
):

    # First pass: discover all task types
    # all_task_types = set()
    # for policy in policies:
    #     for trial in range(1, 4):
    #         dirname = os.path.join(
    #             "../pipeline", "all_logs", f"logs_mofa_{policy}_npipe120000_{trial}"
    #         )
    #         try:
    #             all_task_types.update(get_task_types(dirname, policy, trial))
    #             break
    #         except FileNotFoundError:
    #             continue
    # task_types = sorted(all_task_types)

    max_cpus = 102 * 32
    max_gpus = 12 * 32

    policy_end_times = {
        "Uniform": 1086.45,
        "Partitioned": 1955,
        "Adaptive": 846,
    }

    # 2x2 layout: (0,0) total GPU, (0,1) generate_linker GPU,
    # (1,0) validate_structure GPU, (1,1) optimize_cells GPU
    panels = [
        (0, 0, None, "Total"),
        (0, 1, "generate_linker", "Generate Linkers"),
        (1, 0, "validate_structure", "Validate Structure"),
        (1, 1, "optimize_cells", "Optimize Cells"),
    ]
    fig, axs = get_full_column_fig(nrows=2, ncols=2)

    # Compute global start time for each policy/trial
    global_starts = {}
    for label, policy in zip(labels, policies):
        for trial in range(1, 4):
            dirname = os.path.join(
                "../pipeline",
                "all_logs",
                f"logs_mofa_{policy}_npipe120000_{trial}",
            )
            try:
                global_starts[(policy, trial)] = get_global_start(
                    dirname, policy, trial
                )
                break
            except FileNotFoundError:
                continue

    for r, c, task_type, title in panels:
        ax = axs[r, c]
        for label, policy in zip(labels, policies):
            try:
                times = None
                for trial in range(1, 4):
                    dirname = os.path.join(
                        "../pipeline",
                        "all_logs",
                        f"logs_mofa_{policy}_npipe120000_{trial}",
                    )
                    t0 = global_starts.get((policy, trial))
                    times, cpu_util, gpu_util = compute_utilisation(
                        dirname, policy, trial=trial, task_type=task_type, t0=t0
                    )
                    break
                if times is None:
                    continue
                (line,) = ax.plot(
                    times, gpu_util * 100 / max_gpus, label=label, linestyle="-"
                )
                if task_type is None and label in policy_end_times:
                    ax.axvline(
                        policy_end_times[label],
                        color=line.get_color(),
                        linestyle="--",
                        linewidth=1,
                    )
            except Exception as e:
                print(f"plotting {policy}/{task_type} failed with error {e}")

        if c == 0:
            ax.set_ylabel("GPU Util (%)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    for col in range(2):
        axs[-1, col].set_xlabel("Time (s)")

    handles, lbls = axs[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, lbls, loc="upper center", ncol=len(lbls), bbox_to_anchor=(0.5, 1.08)
    )
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.6)
    fig.savefig("fig_11.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    plot_mofa_utilisation()
