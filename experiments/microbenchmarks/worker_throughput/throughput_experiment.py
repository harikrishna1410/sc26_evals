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


def run_noop_ensemble(ntasks, sleeptime=0.0) -> float:
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
        result_flush_interval=3000,
        result_buffer_size=ntasks,
    )

    el = EnsembleLauncher(
        ensemble_file={}, system_config=sys_config, launcher_config=launcher_config
    )
    el.start()
    time.sleep(10.0)
    with ClusterClient(
        checkpoint_dir=CKPT_DIR,
        n_workers=1,
        task_buffer_size=ntasks,
        task_flush_interval=3000,
    ) as client:
        # futures = [client.submit(echo_hello_world, i, 0.0) for i in range(100)]
        # done, not_done = concurrent.futures.wait(futures)
        tic = time.perf_counter()
        futures = client.submit_batch(list(tasks.values()))
        done, not_done = concurrent.futures.wait(futures)
        toc = time.perf_counter()
    el.stop()
    return toc - tic


def run_noop_ensemble_master(ntasks, sleeptime=0.0) -> float:
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
        policy_config=PolicyConfig(nlevels=1, nchildren=102),
        checkpoint_dir=CKPT_DIR,
        result_flush_interval=0.05,
    )

    el = EnsembleLauncher(
        ensemble_file={}, system_config=sys_config, launcher_config=launcher_config
    )
    el.start()
    time.sleep(10.0)
    with ClusterClient(checkpoint_dir=CKPT_DIR, n_workers=1) as client:
        futures = [client.submit(echo_hello_world, i, 0.0) for i in range(102)]
        done, not_done = concurrent.futures.wait(futures)
        tic = time.perf_counter()
        futures = client.submit_batch(list(tasks.values()))
        done, not_done = concurrent.futures.wait(futures)
        toc = time.perf_counter()
    el.stop()
    return toc - tic


def run_noop_ensemble_batch(ntasks, sleeptime=0.0) -> float:
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
        policy_config=PolicyConfig(nlevels=0),
    )

    el = EnsembleLauncher(
        ensemble_file=tasks, system_config=sys_config, launcher_config=launcher_config
    )

    tic = time.perf_counter()
    results = el.run()
    toc = time.perf_counter()
    el.stop()
    return toc - tic


def run_noop_processpool(
    ntasks, sleeptime=0.0, max_workers=102, warmup_tasks=11
) -> float:
    """
    Run noop tasks using concurrent.futures.ProcessPoolExecutor.

    Args:
        ntasks: Number of tasks to submit
        sleeptime: Time to busy wait in each task (seconds)
        max_workers: Maximum number of worker processes (default: 102)
        warmup_tasks: Number of warmup tasks to run before timing

    Returns:
        Time taken to execute all tasks in seconds
    """
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Warmup phase
        warmup_futures = [
            executor.submit(echo_hello_world, i, sleeptime) for i in range(warmup_tasks)
        ]
        for future in warmup_futures:
            future.result()

        # Actual measurement phase
        start_time = time.perf_counter()
        futures = [
            executor.submit(echo_hello_world, i, sleeptime) for i in range(ntasks)
        ]
        done, not_done = concurrent.futures.wait(futures)
        end_time = time.perf_counter()

    return end_time - start_time


def run_noop_parsl(
    ntasks, sleeptime=0.0, max_workers=102, warmup_tasks=11, use_htex=False
) -> float:
    """
    Run noop tasks using Parsl HighThroughputExecutor.

    Args:
        ntasks: Number of tasks to submit
        sleeptime: Time to busy wait in each task (seconds)
        max_workers: Maximum number of worker processes (default: 102)
        warmup_tasks: Number of warmup tasks to run before timing

    Returns:
        Time taken to execute all tasks in seconds
    """

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
                    max_workers_per_node=max_workers, provider=LocalProvider()
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
        start_time = time.perf_counter()
        futures = [parsl_echo(i, sleeptime) for i in range(ntasks)]
        done, not_done = concurrent.futures.wait(futures)
        end_time = time.perf_counter()
    finally:
        parsl.clear()

    return end_time - start_time


def run_noop_dask(ntasks, sleeptime=0.0, max_workers=102, warmup_tasks=11):
    """
    Run noop tasks using Dask LocalCluster.

    Args:
        ntasks: Number of tasks to submit
        sleeptime: Time to busy wait in each task (seconds)
        max_workers: Maximum number of worker processes (default: 102)
        warmup_tasks: Number of warmup tasks to run before timing

    Returns:
        Time taken to execute all tasks in seconds
    """

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
            start_time = time.perf_counter()
            futures = [
                client.submit(dask_echo_hello_world, i, sleeptime, pure=False)
                for i in range(ntasks)
            ]
            for future in futures:
                future.result()
            end_time = time.perf_counter()

    # Context managers automatically handle proper shutdown:
    # 1. Client closes first (connection to scheduler)
    # 2. Cluster closes last (scheduler and workers)

    return end_time - start_time


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
        choices=[
            "processpool",
            "worker",
            "worker_master",
            "worker_batch",
            "parsl",
            "dask",
            "parsl_htex",
            "all",
        ],
        default="all",
        help="which framework to test",
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
    runtime_processpool = None
    runtime_worker = None
    runtime_worker_master = None
    runtime_worker_batch = None
    runtime_parsl = None
    runtime_dask = None
    runtime_parsl_htex = None
    throughput_processpool = None
    throughput_worker = None
    throughput_worker_master = None
    throughput_worker_batch = None
    throughput_parsl = None
    throughput_dask = None

    run_processpool = args.framework in ("processpool", "all")
    run_worker = args.framework in ("worker", "all")
    run_worker_master = args.framework in ("worker_master", "all")
    run_worker_batch = args.framework in ("worker_batch", "all")
    run_parsl = args.framework in ("parsl", "all")
    run_dask = args.framework in ("dask", "all")
    run_parsl_htex = args.framework in ("parsl_htex", "a;;")

    # Test ProcessPool
    if run_processpool:
        logging.info(
            f"Running ProcessPool with ntasks={args.ntasks}, sleeptime={args.sleeptime}"
        )
        runtime_processpool = run_noop_processpool(args.ntasks, args.sleeptime)
        throughput_processpool = args.ntasks / runtime_processpool
        results.append(
            {
                "method": "processpool",
                "ntasks": args.ntasks,
                "runtime": runtime_processpool,
                "throughput": throughput_processpool,
            }
        )
        logging.info(
            f"ProcessPool - runtime={runtime_processpool:.2f}s, throughput={throughput_processpool:.2f} tasks/s"
        )

    # Test Ensemble Worker
    if run_worker:
        logging.info(
            f"Running Ensemble Worker with ntasks={args.ntasks}, sleeptime={args.sleeptime}"
        )
        runtime_worker = run_noop_ensemble(args.ntasks, args.sleeptime)
        throughput_worker = args.ntasks / runtime_worker
        results.append(
            {
                "method": "worker",
                "ntasks": args.ntasks,
                "runtime": runtime_worker,
                "throughput": throughput_worker,
            }
        )
        logging.info(
            f"Ensemble Worker - runtime={runtime_worker:.2f}s, throughput={throughput_worker:.2f} tasks/s"
        )

    # Test Ensemble Worker Master
    if run_worker_master:
        logging.info(
            f"Running Ensemble Worker Master with ntasks={args.ntasks}, sleeptime={args.sleeptime}"
        )
        runtime_worker_master = run_noop_ensemble_master(args.ntasks, args.sleeptime)
        throughput_worker_master = args.ntasks / runtime_worker_master
        results.append(
            {
                "method": "worker_master",
                "ntasks": args.ntasks,
                "runtime": runtime_worker_master,
                "throughput": throughput_worker_master,
            }
        )
        logging.info(
            f"Ensemble Worker Master - runtime={runtime_worker_master:.2f}s, throughput={throughput_worker_master:.2f} tasks/s"
        )

    # Test Ensemble Worker Batch
    if run_worker_batch:
        logging.info(
            f"Running Ensemble Worker Batch with ntasks={args.ntasks}, sleeptime={args.sleeptime}"
        )
        runtime_worker_batch = run_noop_ensemble_batch(args.ntasks, args.sleeptime)

        throughput_worker_batch = args.ntasks / runtime_worker_batch
        results.append(
            {
                "method": "worker_batch",
                "ntasks": args.ntasks,
                "runtime": runtime_worker_batch,
                "throughput": throughput_worker_batch,
            }
        )
        logging.info(
            f"Ensemble Worker Batch - runtime={runtime_worker_batch:.2f}s, throughput={throughput_worker_batch:.2f} tasks/s"
        )

    # Test Parsl
    if run_parsl:
        logging.info(
            f"Running Parsl with ntasks={args.ntasks}, sleeptime={args.sleeptime}"
        )
        runtime_parsl = run_noop_parsl(args.ntasks, args.sleeptime)
        throughput_parsl = args.ntasks / runtime_parsl
        results.append(
            {
                "method": "parsl",
                "ntasks": args.ntasks,
                "runtime": runtime_parsl,
                "throughput": throughput_parsl,
            }
        )
        logging.info(
            f"Parsl - runtime={runtime_parsl:.2f}s, throughput={throughput_parsl:.2f} tasks/s"
        )

        # Test Parsl
    if run_parsl_htex:
        logging.info(
            f"Running Parsl with ntasks={args.ntasks}, sleeptime={args.sleeptime}"
        )
        runtime_parsl_htex = run_noop_parsl(args.ntasks, args.sleeptime, use_htex=True)
        throughput_parsl_htex = args.ntasks / runtime_parsl_htex
        results.append(
            {
                "method": "parsl_htex",
                "ntasks": args.ntasks,
                "runtime": runtime_parsl_htex,
                "throughput": throughput_parsl_htex,
            }
        )
        logging.info(
            f"Parsl_htex - runtime={runtime_parsl_htex:.2f}s, throughput={throughput_parsl_htex:.2f} tasks/s"
        )

    # Test Dask
    if run_dask:
        logging.info(
            f"Running Dask with ntasks={args.ntasks}, sleeptime={args.sleeptime}"
        )
        runtime_dask = run_noop_dask(args.ntasks, args.sleeptime)
        throughput_dask = args.ntasks / runtime_dask
        results.append(
            {
                "method": "dask",
                "ntasks": args.ntasks,
                "runtime": runtime_dask,
                "throughput": throughput_dask,
            }
        )
        logging.info(
            f"Dask - runtime={runtime_dask:.2f}s, throughput={throughput_dask:.2f} tasks/s"
        )

    # Compare (relative to ProcessPool if available)
    if runtime_processpool is not None:
        if runtime_worker is not None:
            speedup_worker = runtime_worker / runtime_processpool
            logging.info(f"Speedup Worker/ProcessPool: {speedup_worker:.2f}x")
        if runtime_worker_batch is not None:
            speedup_worker_batch = runtime_worker_batch / runtime_processpool
            logging.info(
                f"Speedup Worker Batch/ProcessPool: {speedup_worker_batch:.2f}x"
            )
        if runtime_parsl is not None:
            speedup_parsl = runtime_parsl / runtime_processpool
            logging.info(f"Speedup Parsl/ProcessPool: {speedup_parsl:.2f}x")
        if runtime_dask is not None:
            speedup_dask = runtime_dask / runtime_processpool
            logging.info(f"Speedup Dask/ProcessPool: {speedup_dask:.2f}x")

    # Write Ensemble Worker results to separate CSV
    if runtime_worker is not None:
        worker_output_file = args.output_file.replace(".csv", "_worker.csv")
        file_exists_worker = os.path.isfile(worker_output_file)
        with open(worker_output_file, "a", newline="") as csvfile:
            fieldnames = ["ntasks", "sleeptime", "runtime", "throughput"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists_worker:
                writer.writeheader()
            writer.writerow(
                {
                    "ntasks": args.ntasks,
                    "sleeptime": args.sleeptime,
                    "runtime": runtime_worker,
                    "throughput": throughput_worker,
                }
            )
        logging.info(f"Ensemble Worker results written to {worker_output_file}")

    # Write Ensemble Worker Master results to separate CSV
    if runtime_worker_master is not None:
        worker_master_output_file = args.output_file.replace(
            ".csv", "_worker_master.csv"
        )
        file_exists_worker_master = os.path.isfile(worker_master_output_file)
        with open(worker_master_output_file, "a", newline="") as csvfile:
            fieldnames = ["ntasks", "sleeptime", "runtime", "throughput"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists_worker_master:
                writer.writeheader()
            writer.writerow(
                {
                    "ntasks": args.ntasks,
                    "sleeptime": args.sleeptime,
                    "runtime": runtime_worker_master,
                    "throughput": throughput_worker_master,
                }
            )
        logging.info(
            f"Ensemble Worker Master results written to {worker_master_output_file}"
        )

    # Write Ensemble Worker Batch results to separate CSV
    if runtime_worker_batch is not None:
        worker_batch_output_file = args.output_file.replace(".csv", "_worker_batch.csv")
        file_exists_worker_batch = os.path.isfile(worker_batch_output_file)
        with open(worker_batch_output_file, "a", newline="") as csvfile:
            fieldnames = ["ntasks", "sleeptime", "runtime", "throughput"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists_worker_batch:
                writer.writeheader()
            writer.writerow(
                {
                    "ntasks": args.ntasks,
                    "sleeptime": args.sleeptime,
                    "runtime": runtime_worker_batch,
                    "throughput": throughput_worker_batch,
                }
            )
        logging.info(
            f"Ensemble Worker Batch results written to {worker_batch_output_file}"
        )

    # Write ProcessPool results to CSV
    if runtime_processpool is not None:
        file_exists = os.path.isfile(args.output_file)
        with open(args.output_file, "a", newline="") as csvfile:
            fieldnames = ["ntasks", "sleeptime", "runtime", "throughput"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(
                {
                    "ntasks": args.ntasks,
                    "sleeptime": args.sleeptime,
                    "runtime": runtime_processpool,
                    "throughput": throughput_processpool,
                }
            )
        logging.info(f"ProcessPool results written to {args.output_file}")

    # Write Parsl results to separate CSV
    if runtime_parsl is not None:
        parsl_output_file = args.output_file.replace(".csv", "_parsl.csv")
        file_exists_parsl = os.path.isfile(parsl_output_file)
        with open(parsl_output_file, "a", newline="") as csvfile:
            fieldnames = ["ntasks", "sleeptime", "runtime", "throughput"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists_parsl:
                writer.writeheader()
            writer.writerow(
                {
                    "ntasks": args.ntasks,
                    "sleeptime": args.sleeptime,
                    "runtime": runtime_parsl,
                    "throughput": throughput_parsl,
                }
            )
        logging.info(f"Parsl results written to {parsl_output_file}")

    # Write Parsl results to separate CSV
    if runtime_parsl_htex is not None:
        parsl_htex_output_file = args.output_file.replace(".csv", "_parsl_htex.csv")
        file_exists_parsl_htex = os.path.isfile(parsl_htex_output_file)
        with open(parsl_htex_output_file, "a", newline="") as csvfile:
            fieldnames = ["ntasks", "sleeptime", "runtime", "throughput"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists_parsl_htex:
                writer.writeheader()
            writer.writerow(
                {
                    "ntasks": args.ntasks,
                    "sleeptime": args.sleeptime,
                    "runtime": runtime_parsl_htex,
                    "throughput": throughput_parsl_htex,
                }
            )
        logging.info(f"Parsl results written to {parsl_htex_output_file}")

    # Write Dask results to separate CSV
    if runtime_dask is not None:
        dask_output_file = args.output_file.replace(".csv", "_dask.csv")
        file_exists_dask = os.path.isfile(dask_output_file)
        with open(dask_output_file, "a", newline="") as csvfile:
            fieldnames = ["ntasks", "sleeptime", "runtime", "throughput"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists_dask:
                writer.writeheader()
            writer.writerow(
                {
                    "ntasks": args.ntasks,
                    "sleeptime": args.sleeptime,
                    "runtime": runtime_dask,
                    "throughput": throughput_dask,
                }
            )
        logging.info(f"Dask results written to {dask_output_file}")
