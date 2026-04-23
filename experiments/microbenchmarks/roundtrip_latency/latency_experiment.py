import argparse
import asyncio
import concurrent.futures
import csv
import json
import logging
import os
import shutil
import sys
import time
import uuid
from concurrent.futures import ProcessPoolExecutor

import threading

import parsl
from dask.distributed import Client, LocalCluster
from ensemble_launcher import EnsembleLauncher
from ensemble_launcher.config import LauncherConfig, PolicyConfig, SystemConfig
from ensemble_launcher.ensemble import Task
from ensemble_launcher.helper_functions import get_nodes
from ensemble_launcher.orchestrator import AsyncWorker, ClusterClient
from ensemble_launcher.scheduler.resource import JobResource, NodeResourceList
from parsl import python_app
from parsl.config import Config
from parsl.executors import HighThroughputExecutor, ThreadPoolExecutor
from parsl.providers import LocalProvider

from utils import echo_hello_world


class TaskRegistry:
    """Thread-safe registry to track task submission and completion times"""
    def __init__(self):
        self.lock = threading.Lock()
        self.submissions = {}  # task_id -> submission_time
        self.completions = []  # list of (task_id, completion_time, elapsed_time)
        self.start_time = None
    
    def set_start_time(self):
        """Set the benchmark start time"""
        with self.lock:
            self.start_time = time.time()
    
    def register_submission(self, task_id):
        """Register task submission time"""
        with self.lock:
            self.submissions[task_id] = time.time()
    
    def register_completion(self, task_id):
        """Register task completion time"""
        with self.lock:
            completion_time = time.time()
            # Compute elapsed time from submission
            submission_time = self.submissions.get(task_id, self.start_time)
            elapsed = completion_time - submission_time
            self.completions.append((task_id, submission_time, completion_time, elapsed))
    
    def write_to_csv(self, filename):
        """Write timeline data to CSV file"""
        with self.lock:
            # Sort by completion time
            sorted_completions = sorted(self.completions, key=lambda x: x[1])
            
            with open(filename, 'w') as f:
                f.write("task_id,start_time(s),end_time(s),elapsed_time(s)\n")
                for idx, (task_id, sub_time, comp_time, elapsed) in enumerate(sorted_completions, 1):
                    f.write(f"{task_id},{sub_time:.9f},{comp_time:.9f},{elapsed:.9f}\n")

async def run_noop_ensemble(ntasks, sleeptime=0.0, registry=None, sequential=False) -> tuple:
    if registry is None:
        registry = TaskRegistry()

    tasks = {}
    for i in range(ntasks):
        tasks[f"task_{i}"] = Task(
            task_id=f"task_{i}",
            nnodes=1,
            ppn=1,
            executable=echo_hello_world,
            args=(i, sleeptime),
        )

    CKPT_DIR = f"/tmp/el_ckpt_{uuid.uuid4()}"
    os.makedirs(CKPT_DIR, exist_ok=True)
    ##create the system config
    cpus = list(range(104))
    cpus.pop(52)  # can't use these cores on Aurora
    cpus.pop(0)  # can't use these cores on Aurora
    gpus = list(range(12))
    sys_config = SystemConfig(name="Aurora", cpus=cpus, gpus=gpus)
    nodes = JobResource(
        resources=[NodeResourceList.from_config(sys_config)], nodes=get_nodes()[:1]
    )

    # create launcher config
    launcher_config = LauncherConfig(
        comm_name="async_zmq",
        child_executor_name="async_mpi",
        task_executor_name="async_processpool",
        worker_logs=True,
        master_logs=True,
        return_stdout=True,
        log_level=logging.INFO,
        cluster=True,
        policy_config=PolicyConfig(nlevels=0),
        checkpoint_dir=CKPT_DIR,
        result_buffer_size=0,
        result_flush_interval=10.0,
    )

    el = EnsembleLauncher(
        ensemble_file={}, system_config=sys_config, launcher_config=launcher_config
    )
    el.start()
    time.sleep(10.0)
    with ClusterClient(checkpoint_dir=CKPT_DIR) as client:
        task_ids = list(tasks.keys())
        if sequential:
            for task_id in task_ids:
                registry.register_submission(task_id)
                future = client.submit(tasks[task_id])
                future.result()
                registry.register_completion(task_id)
        else:
            futures = []
            for task_id in task_ids:
                registry.register_submission(task_id)
                future = client.submit(tasks[task_id])
                future.add_done_callback(
                    lambda f, tid=task_id: registry.register_completion(tid)
                )
                futures.append(future)
            _, _ = concurrent.futures.wait(futures)
    el.stop()
    mean_latency = sum(e for _, _, _, e in registry.completions) / len(registry.completions)
    return mean_latency, registry

async def run_noop_ensemble_master(ntasks, sleeptime=0.0, registry=None, sequential=False) -> tuple:
    if registry is None:
        registry = TaskRegistry()

    tasks = {}
    for i in range(ntasks):
        tasks[f"task_{i}"] = Task(
            task_id=f"task_{i}",
            nnodes=1,
            ppn=1,
            executable=echo_hello_world,
            args=(i, sleeptime),
        )

    CKPT_DIR = f"/tmp/el_ckpt_{uuid.uuid4()}"
    os.makedirs(CKPT_DIR, exist_ok=True)
    ##create the system config
    cpus = list(range(104))
    cpus.pop(52)  # can't use these cores on Aurora
    cpus.pop(0)  # can't use these cores on Aurora
    gpus = list(range(12))
    sys_config = SystemConfig(name="Aurora", cpus=cpus, gpus=gpus)
    nodes = JobResource(
        resources=[NodeResourceList.from_config(sys_config)], nodes=get_nodes()[:1]
    )

    # create launcher config
    launcher_config = LauncherConfig(
        comm_name="async_zmq",
        child_executor_name="async_mpi",
        task_executor_name="async_processpool",
        worker_logs=True,
        master_logs=True,
        return_stdout=True,
        log_level=logging.INFO,
        cluster=True,
        policy_config=PolicyConfig(nlevels=1,nchildren=1),
        checkpoint_dir=CKPT_DIR,
        result_buffer_size=0,
        result_flush_interval=10.0,
    )

    el = EnsembleLauncher(
        ensemble_file={}, system_config=sys_config, launcher_config=launcher_config
    )
    el.start()
    time.sleep(10.0)
    with ClusterClient(checkpoint_dir=CKPT_DIR) as client:
        tic = time.perf_counter()
        task_ids = list(tasks.keys())
        if sequential:
            for task_id in task_ids:
                registry.register_submission(task_id)
                future = client.submit(tasks[task_id])
                future.result()
                registry.register_completion(task_id)
        else:
            futures = []
            for task_id in task_ids:
                registry.register_submission(task_id)
                future = client.submit(tasks[task_id])
                future.add_done_callback(
                    lambda f, tid=task_id: registry.register_completion(tid)
                )
                futures.append(future)
            _, _ = concurrent.futures.wait(futures)
        toc = time.perf_counter()
    el.stop()
    mean_latency = sum(e for _, _, _, e in registry.completions) / len(registry.completions)
    return mean_latency, registry


def run_noop_processpool(
    ntasks, sleeptime=0.0, max_workers=102, warmup_tasks=11, registry=None, sequential=False
) -> tuple:
    """
    Run noop tasks using concurrent.futures.ProcessPoolExecutor.

    Args:
        ntasks: Number of tasks to submit
        sleeptime: Time to busy wait in each task (seconds)
        max_workers: Maximum number of worker processes (default: 102)
        warmup_tasks: Number of warmup tasks to run before timing
        registry: TaskRegistry instance for per-task timing (created if None)

    Returns:
        Tuple of (elapsed_time, registry)
    """
    if registry is None:
        registry = TaskRegistry()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Warmup phase
        warmup_futures = [
            executor.submit(echo_hello_world, i, sleeptime) for i in range(warmup_tasks)
        ]
        for future in warmup_futures:
            future.result()

        # Actual measurement phase
        if sequential:
            for i in range(ntasks):
                task_id = f"task_{i}"
                registry.register_submission(task_id)
                future = executor.submit(echo_hello_world, i, sleeptime)
                future.result()
                registry.register_completion(task_id)
        else:
            futures = []
            for i in range(ntasks):
                task_id = f"task_{i}"
                registry.register_submission(task_id)
                future = executor.submit(echo_hello_world, i, sleeptime)
                future.add_done_callback(
                    lambda f, tid=task_id: registry.register_completion(tid)
                )
                futures.append(future)
            concurrent.futures.wait(futures)

    mean_latency = sum(e for _, _, _, e in registry.completions) / len(registry.completions)
    return mean_latency, registry


def run_noop_parsl(
    ntasks, sleeptime=0.0, max_workers=102, warmup_tasks=11, use_htex=False, registry=None, sequential=False
) -> tuple:
    """
    Run noop tasks using Parsl HighThroughputExecutor.

    Args:
        ntasks: Number of tasks to submit
        sleeptime: Time to busy wait in each task (seconds)
        max_workers: Maximum number of worker processes (default: 102)
        warmup_tasks: Number of warmup tasks to run before timing
        registry: TaskRegistry instance for per-task timing (created if None)

    Returns:
        Tuple of (elapsed_time, registry)
    """
    if registry is None:
        registry = TaskRegistry()

    # Define Parsl app
    @python_app
    def parsl_echo(task_id, sleeptime=0.0):
        import time

        if sleeptime > 0:
            start = time.perf_counter()
            while (time.perf_counter() - start) < sleeptime:
                pass  # Busy waiting
        return f"Hello World {task_id}"

    if use_htex:
        # Configure Parsl
        config = Config(
            executors=[
                HighThroughputExecutor(
                    max_workers_per_node=max_workers, provider=LocalProvider(),
                )
            ]
        )
    else:
        # Configure Parsl
        config = Config(executors=[ThreadPoolExecutor(max_threads=max_workers)])

    parsl.load(config)

    try:
        # Warmup phase
        warmup_futures = [parsl_echo(i, sleeptime) for i in range(warmup_tasks)]
        for future in warmup_futures:
            future.result()

        # Actual measurement phase
        if sequential:
            for i in range(ntasks):
                task_id = f"task_{i}"
                registry.register_submission(task_id)
                future = parsl_echo(i, sleeptime)
                future.result()
                registry.register_completion(task_id)
        else:
            futures = []
            for i in range(ntasks):
                task_id = f"task_{i}"
                registry.register_submission(task_id)
                future = parsl_echo(i, sleeptime)
                future.add_done_callback(
                    lambda f, tid=task_id: registry.register_completion(tid)
                )
                futures.append(future)
            concurrent.futures.wait(futures)
    finally:
        parsl.clear()

    mean_latency = sum(e for _, _, _, e in registry.completions) / len(registry.completions)
    return mean_latency, registry


def run_noop_dask(ntasks, sleeptime=0.0, max_workers=102, warmup_tasks=11, registry=None, sequential=False):
    """
    Run noop tasks using Dask LocalCluster.

    Args:
        ntasks: Number of tasks to submit
        sleeptime: Time to busy wait in each task (seconds)
        max_workers: Maximum number of worker processes (default: 102)
        warmup_tasks: Number of warmup tasks to run before timing
        registry: TaskRegistry instance for per-task timing (created if None)

    Returns:
        Tuple of (elapsed_time, registry)
    """
    if registry is None:
        registry = TaskRegistry()

    # Define a wrapper function that imports time in the worker
    def dask_echo_hello_world(task_id, sleeptime=0.0):
        import time  # Import in worker context

        if sleeptime > 0:
            start = time.perf_counter()
            while (time.perf_counter() - start) < sleeptime:
                pass  # Busy waiting
        return f"Hello World {task_id}"

    import dask

    # Increase connection timeouts for large worker counts
    dask.config.set({"distributed.comm.timeouts.connect": "300s"})
    dask.config.set({"distributed.comm.timeouts.tcp": "300s"})

    # Suppress benign shutdown errors from distributed.worker
    logging.getLogger("distributed.worker").setLevel(logging.CRITICAL)
    logging.getLogger("distributed.comm").setLevel(logging.CRITICAL)

    with LocalCluster(
        n_workers=max_workers,
        threads_per_worker=1,
        memory_limit="auto",
        timeout="600s",  # 10 minutes for startup
        silence_logs=logging.CRITICAL,  # Suppress all logs except critical
    ) as cluster:
        with Client(cluster) as client:
            # Ensure the cluster is up before submitting tasks
            client.wait_for_workers(
                n_workers=max_workers, timeout=600
            )  # 10 min timeout

            # Warmup phase
            warmup_futures = [
                client.submit(dask_echo_hello_world, i, sleeptime, pure=False)
                for i in range(warmup_tasks)
            ]
            for future in warmup_futures:
                future.result()

            # Actual measurement phase
            if sequential:
                for i in range(ntasks):
                    task_id = f"task_{i}"
                    registry.register_submission(task_id)
                    future = client.submit(dask_echo_hello_world, i, sleeptime, pure=False)
                    future.result()
                    registry.register_completion(task_id)
            else:
                futures = []
                for i in range(ntasks):
                    task_id = f"task_{i}"
                    registry.register_submission(task_id)
                    future = client.submit(dask_echo_hello_world, i, sleeptime, pure=False)
                    future.add_done_callback(
                        lambda f, tid=task_id: registry.register_completion(tid)
                    )
                    futures.append(future)
                for future in futures:
                    future.result()

    # Context managers automatically handle proper shutdown:
    # 1. Client closes first (connection to scheduler)
    # 2. Cluster closes last (scheduler and workers)

    mean_latency = sum(e for _, _, _, e in registry.completions) / len(registry.completions)
    return mean_latency, registry


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Compare ensemble worker with ProcessPool executor throughput."
    )
    parser.add_argument(
        "--ntasks", type=int, default=102, help="number of tasks to run"
    )
    parser.add_argument(
        "--sleeptime",
        type=float,
        default=0.0,
        help="busy wait time per task in seconds",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="data/throughput_results.csv",
        help="Output CSV file path for results",
    )
    parser.add_argument(
        "--framework",
        type=str,
        choices=["processpool", "worker", "ensemble_master", "parsl", "dask", "parsl_htex", "all"],
        default="all",
        help="which framework to test",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        default=False,
        help="submit one task at a time and wait for each result before submitting the next",
    )

    args = parser.parse_args()

    # Configure logging once at the start
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Store results
    os.makedirs("data", exist_ok=True)

    results = []
    mean_latency_processpool = None
    mean_latency_worker = None
    mean_latency_ensemble_master = None
    mean_latency_parsl = None
    mean_latency_dask = None
    mean_latency_parsl_htex = None

    run_processpool = args.framework in ("processpool", "all")
    run_worker = args.framework in ("worker", "all")
    run_ensemble_master = args.framework in ("ensemble_master", "all")
    run_parsl = args.framework in ("parsl", "all")
    run_dask = args.framework in ("dask", "all")
    run_parsl_htex = args.framework in ("parsl_htex", "a;;")

    # Test ProcessPool
    if run_processpool:
        logging.info(
            f"Running ProcessPool with ntasks={args.ntasks}, sleeptime={args.sleeptime}"
        )
        mean_latency_processpool, registry_processpool = run_noop_processpool(args.ntasks, args.sleeptime, sequential=args.sequential)
        results.append(
            {
                "method": "processpool",
                "ntasks": args.ntasks,
                "mean_latency": mean_latency_processpool,
            }
        )
        logging.info(
            f"ProcessPool - mean_latency={mean_latency_processpool:.6f}s"
        )

    # Test Ensemble Worker
    if run_worker:
        logging.info(
            f"Running Ensemble Worker with ntasks={args.ntasks}, sleeptime={args.sleeptime}"
        )
        mean_latency_worker, registry_worker = asyncio.run(run_noop_ensemble(args.ntasks, args.sleeptime, sequential=args.sequential))
        results.append(
            {
                "method": "worker",
                "ntasks": args.ntasks,
                "mean_latency": mean_latency_worker,
            }
        )
        logging.info(
            f"Ensemble Worker - mean_latency={mean_latency_worker:.6f}s"
        )

    # Test Ensemble Master
    if run_ensemble_master:
        logging.info(
            f"Running Ensemble Master with ntasks={args.ntasks}, sleeptime={args.sleeptime}"
        )
        mean_latency_ensemble_master, registry_ensemble_master = asyncio.run(run_noop_ensemble_master(args.ntasks, args.sleeptime, sequential=args.sequential))
        results.append(
            {
                "method": "ensemble_master",
                "ntasks": args.ntasks,
                "mean_latency": mean_latency_ensemble_master,
            }
        )
        logging.info(
            f"Ensemble Master - mean_latency={mean_latency_ensemble_master:.6f}s"
        )

    # Test Parsl
    if run_parsl:
        logging.info(
            f"Running Parsl with ntasks={args.ntasks}, sleeptime={args.sleeptime}"
        )
        mean_latency_parsl, registry_parsl = run_noop_parsl(args.ntasks, args.sleeptime, sequential=args.sequential)
        results.append(
            {
                "method": "parsl",
                "ntasks": args.ntasks,
                "mean_latency": mean_latency_parsl,
            }
        )
        logging.info(
            f"Parsl - mean_latency={mean_latency_parsl:.6f}s"
        )

    # Test Parsl HTEX
    if run_parsl_htex:
        logging.info(
            f"Running Parsl with ntasks={args.ntasks}, sleeptime={args.sleeptime}"
        )
        mean_latency_parsl_htex, registry_parsl_htex = run_noop_parsl(args.ntasks, args.sleeptime, use_htex=True, sequential=args.sequential)
        results.append(
            {
                "method": "parsl_htex",
                "ntasks": args.ntasks,
                "mean_latency": mean_latency_parsl_htex,
            }
        )
        logging.info(
            f"Parsl_htex - mean_latency={mean_latency_parsl_htex:.6f}s"
        )

    # Test Dask
    if run_dask:
        logging.info(
            f"Running Dask with ntasks={args.ntasks}, sleeptime={args.sleeptime}"
        )
        mean_latency_dask, registry_dask = run_noop_dask(args.ntasks, args.sleeptime, sequential=args.sequential)
        results.append(
            {
                "method": "dask",
                "ntasks": args.ntasks,
                "mean_latency": mean_latency_dask,
            }
        )
        logging.info(
            f"Dask - mean_latency={mean_latency_dask:.6f}s"
        )

    # Compare (relative to ProcessPool if available)
    if mean_latency_processpool is not None:
        if mean_latency_worker is not None:
            ratio_worker = mean_latency_worker / mean_latency_processpool
            logging.info(f"Latency ratio Worker/ProcessPool: {ratio_worker:.2f}x")
        if mean_latency_ensemble_master is not None:
            ratio_ensemble_master = mean_latency_ensemble_master / mean_latency_processpool
            logging.info(f"Latency ratio EnsembleMaster/ProcessPool: {ratio_ensemble_master:.2f}x")
        if mean_latency_parsl is not None:
            ratio_parsl = mean_latency_parsl / mean_latency_processpool
            logging.info(f"Latency ratio Parsl/ProcessPool: {ratio_parsl:.2f}x")
        if mean_latency_dask is not None:
            ratio_dask = mean_latency_dask / mean_latency_processpool
            logging.info(f"Latency ratio Dask/ProcessPool: {ratio_dask:.2f}x")

    # Write Ensemble Worker results to separate CSV
    if mean_latency_worker is not None:
        worker_output_file = args.output_file.replace(".csv", "_worker.csv")
        file_exists_worker = os.path.isfile(worker_output_file)
        with open(worker_output_file, "a", newline="") as csvfile:
            fieldnames = ["ntasks", "sleeptime", "mean_latency"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists_worker:
                writer.writeheader()
            writer.writerow(
                {
                    "ntasks": args.ntasks,
                    "sleeptime": args.sleeptime,
                    "mean_latency": mean_latency_worker,
                }
            )
        logging.info(f"Ensemble Worker results written to {worker_output_file}")
        worker_tasks_file = args.output_file.replace(".csv", "_worker_tasks.csv")
        registry_worker.write_to_csv(worker_tasks_file)
        logging.info(f"Ensemble Worker per-task timings written to {worker_tasks_file}")

    # Write Ensemble Master results to separate CSV
    if mean_latency_ensemble_master is not None:
        ensemble_master_output_file = args.output_file.replace(".csv", "_ensemble_master.csv")
        file_exists_ensemble_master = os.path.isfile(ensemble_master_output_file)
        with open(ensemble_master_output_file, "a", newline="") as csvfile:
            fieldnames = ["ntasks", "sleeptime", "mean_latency"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists_ensemble_master:
                writer.writeheader()
            writer.writerow(
                {
                    "ntasks": args.ntasks,
                    "sleeptime": args.sleeptime,
                    "mean_latency": mean_latency_ensemble_master,
                }
            )
        logging.info(f"Ensemble Master results written to {ensemble_master_output_file}")
        ensemble_master_tasks_file = args.output_file.replace(".csv", "_ensemble_master_tasks.csv")
        registry_ensemble_master.write_to_csv(ensemble_master_tasks_file)
        logging.info(f"Ensemble Master per-task timings written to {ensemble_master_tasks_file}")

    # Write ProcessPool results to CSV
    if mean_latency_processpool is not None:
        file_exists = os.path.isfile(args.output_file)
        with open(args.output_file, "a", newline="") as csvfile:
            fieldnames = ["ntasks", "sleeptime", "mean_latency"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(
                {
                    "ntasks": args.ntasks,
                    "sleeptime": args.sleeptime,
                    "mean_latency": mean_latency_processpool,
                }
            )
        logging.info(f"ProcessPool results written to {args.output_file}")
        processpool_tasks_file = args.output_file.replace(".csv", "_processpool_tasks.csv")
        registry_processpool.write_to_csv(processpool_tasks_file)
        logging.info(f"ProcessPool per-task timings written to {processpool_tasks_file}")

    # Write Parsl results to separate CSV
    if mean_latency_parsl is not None:
        parsl_output_file = args.output_file.replace(".csv", "_parsl.csv")
        file_exists_parsl = os.path.isfile(parsl_output_file)
        with open(parsl_output_file, "a", newline="") as csvfile:
            fieldnames = ["ntasks", "sleeptime", "mean_latency"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists_parsl:
                writer.writeheader()
            writer.writerow(
                {
                    "ntasks": args.ntasks,
                    "sleeptime": args.sleeptime,
                    "mean_latency": mean_latency_parsl,
                }
            )
        logging.info(f"Parsl results written to {parsl_output_file}")
        parsl_tasks_file = args.output_file.replace(".csv", "_parsl_tasks.csv")
        registry_parsl.write_to_csv(parsl_tasks_file)
        logging.info(f"Parsl per-task timings written to {parsl_tasks_file}")

    # Write Parsl HTEX results to separate CSV
    if mean_latency_parsl_htex is not None:
        parsl_htex_output_file = args.output_file.replace(".csv", "_parsl_htex.csv")
        file_exists_parsl_htex = os.path.isfile(parsl_htex_output_file)
        with open(parsl_htex_output_file, "a", newline="") as csvfile:
            fieldnames = ["ntasks", "sleeptime", "mean_latency"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists_parsl_htex:
                writer.writeheader()
            writer.writerow(
                {
                    "ntasks": args.ntasks,
                    "sleeptime": args.sleeptime,
                    "mean_latency": mean_latency_parsl_htex,
                }
            )
        logging.info(f"Parsl HTEX results written to {parsl_htex_output_file}")
        parsl_htex_tasks_file = args.output_file.replace(".csv", "_parsl_htex_tasks.csv")
        registry_parsl_htex.write_to_csv(parsl_htex_tasks_file)
        logging.info(f"Parsl HTEX per-task timings written to {parsl_htex_tasks_file}")

    # Write Dask results to separate CSV
    if mean_latency_dask is not None:
        dask_output_file = args.output_file.replace(".csv", "_dask.csv")
        file_exists_dask = os.path.isfile(dask_output_file)
        with open(dask_output_file, "a", newline="") as csvfile:
            fieldnames = ["ntasks", "sleeptime", "mean_latency"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists_dask:
                writer.writeheader()
            writer.writerow(
                {
                    "ntasks": args.ntasks,
                    "sleeptime": args.sleeptime,
                    "mean_latency": mean_latency_dask,
                }
            )
        logging.info(f"Dask results written to {dask_output_file}")
        dask_tasks_file = args.output_file.replace(".csv", "_dask_tasks.csv")
        registry_dask.write_to_csv(dask_tasks_file)
        logging.info(f"Dask per-task timings written to {dask_tasks_file}")
