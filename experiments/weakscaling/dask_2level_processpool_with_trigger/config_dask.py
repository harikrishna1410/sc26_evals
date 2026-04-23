"""
Configuration helpers for setting up Dask distributed workers on Aurora HPC system.

This module provides utilities to launch Dask workers with proper CPU affinity
and GPU resource configuration for different execution patterns.
"""

import logging
import os
import random
import subprocess
from typing import Dict, List, Optional

from dask.distributed import WorkerPlugin
from loky import get_reusable_executor


class AddProcessPool(WorkerPlugin):
    def setup(self, worker):
        """Create a ProcessPoolExecutor for the worker."""
        try:
            executor = get_reusable_executor(max_workers=worker.nthreads)
            worker.executors["processes"] = executor  # Named executor, not default
            worker._custom_executor = executor  # Keep reference for cleanup
        except Exception as e:
            worker.log_event("custom-executor", {"error": str(e)})
            raise

    def teardown(self, worker):
        """Shut down the ProcessPoolExecutor when worker stops."""
        if hasattr(worker, "_custom_executor"):
            worker._custom_executor.shutdown(wait=True)
            del worker._custom_executor


def get_num_nodes() -> int:
    """Get the number of nodes from PBS_NODEFILE."""
    node_file = os.getenv("PBS_NODEFILE")
    with open(node_file, "r") as f:
        node_list = f.readlines()
        return len(node_list)


def get_node_list() -> List[str]:
    """Get the list of node hostnames from PBS_NODEFILE."""
    node_file = os.getenv("PBS_NODEFILE")
    with open(node_file, "r") as f:
        return [line.strip() for line in f.readlines()]


def get_scheduler_file() -> str:
    """Get path to scheduler file."""
    return os.path.join(os.getcwd(), "scheduler_file.json")


def get_worker_options(
    worker_type: str,
    n_workers: int,
    cpu_affinity: Optional[str] = None,
    gpu_list: Optional[List[int]] = None,
    memory_limit: str = "auto",
    nthreads: int = 1,
) -> Dict:
    """
    Generate worker configuration options.

    Args:
        worker_type: Type of worker ("cpu" or "gpu")
        n_workers: Number of workers
        cpu_affinity: CPU affinity string (e.g., "list:1:2:3")
        gpu_list: List of GPU indices to assign to workers
        memory_limit: Memory limit per worker
        nthreads: Number of threads per worker

    Returns:
        Dictionary of worker options
    """
    options = {
        "nworkers": n_workers,
        "nthreads": nthreads,
        "memory_limit": memory_limit,
        "name": worker_type,
    }

    if cpu_affinity is not None:
        options["cpu_affinity"] = cpu_affinity

    if gpu_list is not None:
        options["gpu_list"] = gpu_list

    return options


class DaskWorkerConfig:
    """Configuration class for Dask workers on Aurora."""

    def __init__(self):
        self.num_nodes = get_num_nodes()
        self.node_list = get_node_list()

        # Aurora CPU configuration
        self.all_cores = [f"{i}" for i in range(104)]
        self.all_cores.remove("52")  # Remove core 52
        self.all_cores.remove("0")  # Remove core 0

        # Start cores for GPU tiles
        self.start_cores = [1, 9, 17, 25, 33, 41, 53, 61, 69, 77, 85, 93]

    def get_single_core_tile_config(self) -> Dict:
        """
        Configuration for 102 CPU workers + 12 GPU workers per node.
        Each worker uses a single core/tile.
        """
        cpu_affinity_cpu = "list:" + ":".join(self.all_cores)
        cpu_affinity_gpu = "list:" + ":".join(
            [f"{st}-{st + 7}" for st in self.start_cores]
        )

        return {
            "cpu": get_worker_options(
                worker_type="cpu",
                n_workers=102,
                cpu_affinity=cpu_affinity_cpu,
            ),
            "gpu": get_worker_options(
                worker_type="gpu",
                n_workers=12,
                cpu_affinity=cpu_affinity_gpu,
                gpu_list=[i for i in range(12)],
            ),
        }

    def get_single_core_config(self) -> Dict:
        """
        Configuration for 102 CPU workers per node only.
        """
        cpu_affinity = "list:" + ":".join(self.all_cores)

        return {
            "cpu": get_worker_options(
                worker_type="cpu",
                n_workers=102,
                cpu_affinity=cpu_affinity,
            ),
        }

    def get_single_tile_config(self) -> Dict:
        """
        Configuration for 12 GPU workers per node (1 worker per tile).
        """
        cpu_affinity = "list:" + ":".join([f"{st}-{st + 7}" for st in self.start_cores])

        return {
            "gpu": get_worker_options(
                worker_type="gpu",
                n_workers=12,
                cpu_affinity=cpu_affinity,
                gpu_list=[i for i in range(12)],
            ),
        }

    def get_two_ccs_config(self) -> Dict:
        """
        Configuration for 24 GPU workers per node (2 workers per tile).
        """
        cpu_affinity = "list:" + ":".join(
            [f"{st}-{st + 3}:{st + 4}-{st + 7}" for st in self.start_cores]
        )
        gpu_list = [i for i in range(12) for _ in range(2)]

        return {
            "gpu": get_worker_options(
                worker_type="gpu",
                n_workers=24,
                cpu_affinity=cpu_affinity,
                gpu_list=gpu_list,
            ),
        }

    def get_four_ccs_config(self) -> Dict:
        """
        Configuration for 48 GPU workers per node (4 workers per tile).
        """
        cpu_affinity = "list:" + ":".join(
            [
                f"{st}-{st + 1}:{st + 2}-{st + 3}:{st + 4}-{st + 5}:{st + 6}-{st + 7}"
                for st in self.start_cores
            ]
        )
        gpu_list = [i for i in range(12) for _ in range(4)]

        return {
            "gpu": get_worker_options(
                worker_type="gpu",
                n_workers=48,
                cpu_affinity=cpu_affinity,
                gpu_list=gpu_list,
            ),
        }

    def get_any_worker_config(self, nworkers: int = 64) -> Dict:
        """
        Configuration for 48 GPU workers per node (4 workers per tile).
        """
        assert nworkers <= len(self.all_cores)
        cpu_affinity = "list:" + ":".join([f"{st}" for st in self.all_cores[:nworkers]])

        return {
            "gpu": get_worker_options(
                worker_type="cpu",
                n_workers=nworkers,
                cpu_affinity=cpu_affinity,
            ),
        }


def launch_scheduler(
    interface: str = "bond0",
    base_port: int = 8786,
    logger: Optional[logging.Logger] = None,
) -> subprocess.Popen:
    """
    Launch a Dask scheduler.

    Args:
        interface: Network interface to bind to
        port: Port number for scheduler
        logger: Optional logger instance for logging

    Returns:
        Popen object for the scheduler process
    """
    scheduler_file = get_scheduler_file()
    cmd = [
        "dask-scheduler",
        f"--interface={interface}",
        f"--port={base_port + int(random.uniform(1, 1000))}",
        f"--scheduler-file={scheduler_file}",
    ]

    # Remove scheduler file if it exists
    if os.path.exists(scheduler_file):
        os.remove(scheduler_file)

    if logger:
        logger.info(f"Launching scheduler: {' '.join(cmd)}")

    proc = subprocess.Popen(cmd)

    # Wait for scheduler file to be created
    import time

    max_wait = 30
    waited = 0
    while not os.path.exists(scheduler_file) and waited < max_wait:
        time.sleep(0.5)
        waited += 0.5

    if not os.path.exists(scheduler_file):
        raise RuntimeError(f"Scheduler file not created after {max_wait} seconds")

    if logger:
        logger.info(f"Scheduler file created: {scheduler_file}")
    return proc


def launch_workers_with_config(
    config_dict: Dict,
    interface: str = "bond0",
    logger: Optional[logging.Logger] = None,
    one_worker_per_node: bool = False,
):
    """
    Launch Dask workers based on configuration dictionary.

    Args:
        config_dict: Dictionary of worker configurations (from DaskWorkerConfig methods)
        interface: Network interface to bind to
        logger: Optional logger instance for logging
        one_worker_per_node: If True, launch 1 worker per node with N threads.
                            If False, launch N workers per node with 1 thread each.

    Returns:
        List of tuples containing (process, stdout_file, stderr_file) for each worker type
    """
    scheduler_file = get_scheduler_file()
    num_nodes = get_num_nodes()
    worker_procs = []

    for worker_type, worker_opts in config_dict.items():
        nworkers_config = worker_opts["nworkers"]  # From config: number to use
        nthreads = worker_opts.get("nthreads", 1)
        resources = worker_opts.get("resources", {})
        cpu_affinity = worker_opts.get("cpu_affinity", "list:1")

        # Build CPU affinity string - convert from "list:1:2:3" to "1,2,3" for MPI
        if cpu_affinity.startswith("list:"):
            affinity_list = cpu_affinity[5:].replace(":", ",")
        else:
            affinity_list = "1"

        if one_worker_per_node:
            # Mode 1: 1 worker per node with nworkers_config threads
            total_workers = num_nodes
            ppn = 1
            nthreads_per_worker = nworkers_config
            if logger:
                logger.info(
                    f"Launching {total_workers} {worker_type} workers (1 per node, {nthreads_per_worker} threads each)"
                )
        else:
            # Mode 2: nworkers_config workers per node with 1 thread each
            total_workers = nworkers_config * num_nodes
            ppn = nworkers_config
            nthreads_per_worker = 1
            if logger:
                logger.info(
                    f"Launching {total_workers} {worker_type} workers ({ppn} per node, 1 thread each)"
                )

        # Build resource string for dask-worker
        resource_str = ",".join([f"{k}={v}" for k, v in resources.items()])

        if "gpu_list" in config_dict:
            resource_str = resource_str + "," + f"gpu={len(config_dict['gpu_list'])}"

        cmd = [
            "mpiexec",
            "-n",
            str(total_workers),
            "--ppn",
            str(ppn),
            "--cpu-bind",
            f"list:{affinity_list}",
            "dask-worker",
            f"--scheduler-file={scheduler_file}",
            f"--interface={interface}",
            f"--nthreads={nthreads_per_worker}",
            "--no-dashboard",
            "--no-nanny",  # Add this line
        ]

        if resource_str:
            cmd.extend(["--resources", resource_str])

        if logger:
            logger.info(f"Worker command: {' '.join(cmd)}")

        # Launch workers and redirect output to log files for debugging
        worker_stdout = open(f"logs/worker_{worker_type}_stdout.log", "w")
        worker_stderr = open(f"logs/worker_{worker_type}_stderr.log", "w")
        proc = subprocess.Popen(cmd, stdout=worker_stdout, stderr=worker_stderr)
        worker_procs.append((proc, worker_stdout, worker_stderr))

    return worker_procs
