import argparse
import os
import subprocess

os.environ["EL_EXTERNAL_POLICY_PATH"] = os.path.join(os.getcwd(), "policies")
os.environ["EL_EXTERNAL_POLICY_MODULE"] = "custom_policies"
import random
import time
from typing import List, Tuple

import numpy as np
from ensemble_launcher import EnsembleLauncher
from ensemble_launcher.config import (
    LauncherConfig,
    MPIConfig,
    PolicyConfig,
    SystemConfig,
)
from ensemble_launcher.ensemble import Task
from ensemble_launcher.logging import setup_logger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntasks", type=int, default=1000, help="Number of tasks")
    parser.add_argument("--nnodes", type=str, default="1,2", help="Node range")
    parser.add_argument(
        "--mean_duration_sec", type=float, default=30.0, help="Sleep time in seconds"
    )
    parser.add_argument(
        "--variance_sec",
        type=float,
        default=1.0,
        help="Variance of sleep time in seconds",
    )
    parser.add_argument("--policy", type=str, default="fifo_policy", help="policy name")
    args = parser.parse_args()

    return args


def generate_node_count_duration(
    node_count_range: Tuple = (1, 4),
    mean_duration_sec: float = 30.0,
    variance_sec: float = 1.0,
    ntasks: int = 1000,
) -> List[Tuple[float, float]]:
    nd = []
    alpha = mean_duration_sec**2 / variance_sec
    beta = variance_sec / mean_duration_sec
    for i in range(ntasks):
        ##sample a node count from uniform distribution
        node_count = random.randint(*node_count_range)
        ##sample using gamma distribution
        duration = random.gammavariate(alpha=alpha, beta=beta)
        nd.append((node_count, duration))

    return nd


def run_bag_of_tasks(args):
    logger = setup_logger(name="script", log_dir="logs")
    # get the num_nodes and duration
    nd = generate_node_count_duration(
        ntasks=args.ntasks,
        node_count_range=tuple(map(int, args.nnodes.split(","))),
        mean_duration_sec=args.mean_duration_sec,
        variance_sec=args.variance_sec,
    )

    mean = tuple(np.mean(np.array(nd), axis=0))
    std = tuple(np.std(np.array(nd), axis=0))
    logger.info(f"(nnodes, dur) mean:{mean} std:{std}")
    tasks = {}
    cwd = os.getcwd()
    for idx, (nnodes, duration) in enumerate(nd):
        task_id = f"task_{idx}"
        tasks[task_id] = Task(
            task_id=task_id,
            nnodes=nnodes,
            ppn=12,
            ngpus_per_process=1,
            executable=f"python3 {cwd}/mpi_example.py --task_id {idx} --sleep_time {duration}",
            estimated_runtime=duration,
        )

    # Read PBS_NODEFILE to determine number of nodes
    nodefile = os.environ.get("PBS_NODEFILE", "")
    if nodefile and os.path.isfile(nodefile):
        with open(nodefile, "r") as f:
            nnodes_total = len(f.readlines())
    else:
        nnodes_total = 1

    for policy in args.policy.split(","):
        logger.info(f"*****************Running {policy} policy******************")

        # Update task executables to include policy
        for task in tasks.values():
            task.executable = (
                f"python3 {cwd}/mpi_example.py"
                f" --task_id {task.task_id.split('_')[1]}"
                f" --sleep_time {task.estimated_runtime}"
                f" --policy {policy}"
                f" --nnodes {task.nnodes}"
            )

        launcher_config = LauncherConfig(
            task_executor_name="async_mpi",
            worker_logs=True,
            task_scheduler_policy=policy,
            policy_config=PolicyConfig(nlevels=0, strict_priority=True),
            mpi_config=MPIConfig(flavor="mpich"),
        )

        cpus = list(range(104))
        cpus.pop(52)
        cpus.pop(0)
        gpus = list(range(12))

        sys_config = SystemConfig(
            name="aurora", ncpus=102, cpus=cpus, gpus=gpus, ngpus=12
        )

        el = EnsembleLauncher(
            tasks, system_config=sys_config, launcher_config=launcher_config
        )

        logger.info("Staring execution")
        tic = time.perf_counter()
        result = el.run()
        toc = time.perf_counter()
        logger.info(f"Done execution. Total execution time: {toc - tic}")

        # Merge timelines across nodes
        logger.info(f"Merging timelines for policy {policy}")
        merge_cmd = [
            "mpirun",
            "-np", str(nnodes_total),
            "--ppn", "1",
            "python3", f"{cwd}/merge_timelines.py",
            "--policy", policy,
        ]
        merge_proc = subprocess.Popen(merge_cmd)
        merge_proc.wait()
        logger.info(f"Timeline merge complete for policy {policy}, rc={merge_proc.returncode}")

    return result.to_dict()


if __name__ == "__main__":
    args = parse_args()
    run_bag_of_tasks(args)
