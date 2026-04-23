import argparse
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import TimeoutError

import dask
from config_dask import (
    AddProcessPool,
    DaskWorkerConfig,
    get_scheduler_file,
    launch_scheduler,
    launch_workers_with_config,
)
from dask.distributed import Client, as_completed, wait
from utils import sleep_task

dask.config.set({"distributed.comm.timeouts.connect": "120s"})
dask.config.set({"distributed.comm.timeouts.tcp": "120s"})
dask.config.set({"distributed.comm.retry.count": 10})


class TaskRegistry:
    """Thread-safe registry to track task submission and completion times"""

    def __init__(self):
        self.lock = threading.Lock()
        self.submissions = {}  # task_id -> submission_time
        self.completions = []  # list of (task_id, submission_time, completion_time, elapsed_time)
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
            self.completions.append(
                (task_id, submission_time, completion_time, elapsed)
            )

    def write_to_csv(self, filename):
        """Write timeline data to CSV file"""
        with self.lock:
            # Sort by completion time
            sorted_completions = sorted(self.completions, key=lambda x: x[1])

            with open(filename, "w") as f:
                f.write("task_id,start_time(s),end_time(s),elapsed_time(s)\n")
                for idx, (task_id, sub_time, comp_time, elapsed) in enumerate(
                    sorted_completions, 1
                ):
                    f.write(f"{task_id},{sub_time:.9f},{comp_time:.9f},{elapsed:.9f}\n")


# Setup logging
# Set dask logging to only show critical errors
logging.getLogger("distributed").setLevel(logging.CRITICAL)
logging.getLogger("dask").setLevel(logging.CRITICAL)
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("logs/main.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def trigger_task(trigger_file):
    """Trigger task that waits for a file to be written"""
    import os
    import time

    while not os.path.exists(trigger_file):
        time.sleep(0.001)  # Check every 1ms
    return True


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sleep-time", type=float, default=10.0, help="sleep time for tasks in seconds"
    )
    parser.add_argument(
        "--num-tasks", type=int, default=10, help="total number of tasks to run"
    )
    parser.add_argument(
        "--concurrent-workers",
        type=int,
        default=64,
        help="number of concurrent workers",
    )
    parser.add_argument(
        "--scheduler-mode",
        type=str,
        default="manual",
        choices=["manual", "mpi"],
        help="how to launch scheduler and workers (manual or mpi)",
    )
    parser.add_argument(
        "--track-timeline",
        action="store_true",
        help="track and log detailed task timeline",
    )
    parser.add_argument(
        "--one-worker-per-node",
        action="store_true",
        help="launch 1 worker per node with N threads (default: N workers per node with 1 thread)",
    )
    parser.add_argument(
        "--processpool",
        action="store_true",
        help="Use a processpool instead of threadpool for launching worker",
    )
    parser.add_argument(
        "--trigger",
        action="store_true",
        help="Use dask.delayed to trigger all tasks after submission (default: immediate submission)",
    )
    return parser.parse_args()


def wait_for_tasks_with_timeline(
    futures, zero_latency_runtime, timeout_multiplier=50, registry=None
):
    """Wait for all tasks to finish and return completion statistics with optional timeline tracking"""
    n_tasks = len(futures)
    logger.info(f"Waiting for {n_tasks} tasks to finish")

    start_time = time.perf_counter()
    n_finished = 0
    timeout = zero_latency_runtime * timeout_multiplier + 300.0

    if registry:
        # Track completion timeline using registry callbacks
        # Just wait for all futures to complete
        try:
            for future in as_completed(futures, timeout=timeout):
                n_finished += 1

                # Log progress at intervals
                if n_finished % max(1, n_tasks // 10) == 0:
                    elapsed = time.perf_counter() - start_time
                    logger.info(
                        f"{n_finished}/{n_tasks} tasks completed ({elapsed:.2f}s)"
                    )
        except TimeoutError:
            logger.warning("Some tasks timed out")
    else:
        # Original behavior - just wait without tracking individual completions
        try:
            wait(futures, timeout=timeout)
            n_finished = sum(1 for f in futures if f.done())
        except TimeoutError:
            logger.warning("Some tasks timed out")
            n_finished = sum(1 for f in futures if f.done())

    end_time = time.perf_counter()
    runtime = end_time - start_time

    return n_finished, runtime


def run_tasks_no_wait(
    client,
    n_tasks,
    sleep_time,
    worker_resources=None,
    registry=None,
    use_processpool=False,
    use_trigger=False,
):
    """Submit tasks using either Dask Delayed (triggered) or direct submission

    Args:
        n_tasks: Total number of tasks to run
        registry: Optional TaskRegistry for timeline tracking
        use_processpool: If True, use the 'processes' executor via dask.annotate
        use_trigger: If True, use dask.delayed to batch submit all tasks before execution
    """

    logger.info(
        f"Running {n_tasks} tasks"
        + (" with processpool" if use_processpool else "")
        + (" with trigger (delayed)" if use_trigger else " with immediate submission")
    )

    # Use context manager for processpool if requested
    submit_context = (
        dask.annotate(executor="processes") if use_processpool else dask.annotate()
    )

    if use_trigger:
        # DAG-based trigger mode: create a trigger task that all other tasks depend on
        trigger_file = "dask_trigger_file"
        # Remove old trigger file if exists
        if os.path.exists(trigger_file):
            os.remove(trigger_file)

        # Submit trigger task that waits for file
        trigger_future = client.submit(trigger_task, trigger_file, pure=False)

        futures = []
        with submit_context:
            for i in range(n_tasks):
                if registry:
                    registry.register_submission(i)

                # Submit task with dependency on trigger_future
                def dependent_task(trigger_result, task_id, sleep_time):
                    # trigger_result ensures this runs after trigger completes
                    return sleep_task(task_id, sleep_time)

                if worker_resources:
                    future = client.submit(
                        dependent_task,
                        trigger_future,  # Dependency on trigger
                        i,
                        sleep_time,
                        resources=worker_resources,
                        pure=False,
                    )
                else:
                    future = client.submit(
                        dependent_task,
                        trigger_future,  # Dependency on trigger
                        i,
                        sleep_time,
                        pure=False,
                    )

                # Add done callback if tracking timeline
                if registry:
                    future.add_done_callback(
                        lambda fut, tid=i: registry.register_completion(tid)
                    )

                futures.append(future)

        # Return both trigger file path and futures
        return (trigger_file, futures)
    else:
        # Direct submission mode (original behavior)
        futures = []

        with submit_context:
            for i in range(n_tasks):
                if registry:
                    registry.register_submission(i)

                # Submit task with resource specification if provided
                if worker_resources:
                    future = client.submit(
                        sleep_task,
                        i,
                        sleep_time,
                        resources=worker_resources,
                        pure=False,
                    )
                else:
                    future = client.submit(sleep_task, i, sleep_time, pure=False)

                # Add done callback if tracking timeline
                if registry:
                    future.add_done_callback(
                        lambda fut, tid=i: registry.register_completion(tid)
                    )

                futures.append(future)

    return futures


def run_tasks(
    client,
    config_name,
    n_tasks,
    sleep_time,
    n_concurrent_slots_per_node,
    n_nodes,
    worker_resources=None,
    track_timeline=False,
    use_processpool=False,
    use_trigger=False,
):
    """Run tasks with the given configuration and parameters

    Args:
        n_tasks: Total number of tasks to run
        n_concurrent_slots_per_node: Number of concurrent task slots per node (threads per worker)
        n_nodes: Number of nodes
        track_timeline: Whether to track and log detailed task timeline
        use_processpool: If True, use the 'processes' executor
        use_trigger: If True, use dask.delayed for triggered execution
    """

    logger.info(f"Hello Dask ({config_name})")

    import math

    n_concurrent_slots_total = n_concurrent_slots_per_node * n_nodes
    zero_latency_runtime = sleep_time * math.ceil(n_tasks / n_concurrent_slots_total)

    # Create registry if tracking timeline
    registry = TaskRegistry() if track_timeline else None
    if registry:
        registry.set_start_time()

    # Record submission start time
    submission_start = time.perf_counter()

    result = run_tasks_no_wait(
        client,
        n_tasks,
        sleep_time,
        worker_resources,
        registry=registry,
        use_processpool=use_processpool,
        use_trigger=use_trigger,
    )

    # Handle trigger mode vs regular mode
    if use_trigger:
        trigger_file, futures = result
    else:
        futures = result

    submission_time = time.perf_counter() - submission_start
    logger.info(f"Task submission completed in {submission_time:.2f} seconds")

    # If using trigger mode, write the trigger file to start execution
    if use_trigger:
        trigger_timestamp = time.time()
        logger.info(f"Writing trigger file at timestamp {trigger_timestamp:.9f}...")
        temp_file = trigger_file + ".tmp"
        with open(temp_file, "w") as f:
            f.write(f"{trigger_timestamp}\n")
        os.rename(temp_file, trigger_file)  # Atomic rename
        logger.info("Trigger file written - tasks will now execute")

    # Wait for all tasks to finish with optional timeline tracking
    n_finished, runtime = wait_for_tasks_with_timeline(
        futures, zero_latency_runtime, registry=registry
    )

    # Write timeline to CSV if tracking
    if registry:
        timeline_file = "logs/timeline_client.csv"
        registry.write_to_csv(timeline_file)
        logger.info(f"Timeline written to {timeline_file}")

    if n_finished == n_tasks:
        logger.info(f"All tasks finished in {runtime:.2f} seconds")
        logger.info(f"Tasks per second per node: {n_tasks / runtime / n_nodes:.2f}")
        logger.info(
            f"Runtime - Zero latency runtime: {runtime - zero_latency_runtime:.2f} seconds"
        )
    else:
        logger.warning(
            f"Not all tasks finished, only {n_finished} out of {n_tasks} tasks"
        )
        logger.info(f"Runtime is {runtime:.2f} seconds")


def cleanup_workers(worker_procs, scheduler_proc):
    """Clean up worker and scheduler processes"""
    logger.info("Cleaning up workers and scheduler...")
    for item in worker_procs:
        if isinstance(item, tuple):
            proc, stdout, stderr = item
            stdout.close()
            stderr.close()
        else:
            proc = item
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    nodefile = os.environ.get("PBS_NODEFILE", "/dev/null")
    with open(nodefile, "r") as f:
        nodes = [line.split(".")[0] for line in f.readlines()]

    for node in nodes:
        try:
            subprocess.run(["ssh", node, "pkill", "-f", "dask-worker"], timeout=10)
        except subprocess.TimeoutExpired:
            logger.warning(f"pkill timed out on {node}")

    if scheduler_proc:
        scheduler_proc.terminate()
        try:
            scheduler_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            scheduler_proc.kill()


def get_worker_config(args):
    """Get worker configuration based on concurrent workers argument"""
    worker_config = DaskWorkerConfig()

    # if args.concurrent_workers == 12:
    #     config_dict = worker_config.get_single_tile_config()
    #     config_name = "tile config"
    # elif args.concurrent_workers == 24:
    #     config_dict = worker_config.get_two_ccs_config()
    #     config_name = "2 ccs config"
    # elif args.concurrent_workers == 48:
    #     config_dict = worker_config.get_four_ccs_config()
    #     config_name = "4 ccs config"
    # elif args.concurrent_workers == 102:
    #     config_dict = worker_config.get_single_core_tile_config()
    #     config_name = "full node config"
    # else:
    config_dict = worker_config.get_any_worker_config(nworkers=args.concurrent_workers)
    config_name = f"{args.concurrent_workers} config"

    return config_dict, config_name


def initialize_dask_client(args):
    """Initialize Dask client in either MPI or manual mode

    Returns:
        tuple: (client, scheduler_proc, worker_procs, config_dict)
    """
    scheduler_proc = None
    worker_procs = []
    config_dict = None

    if args.scheduler_mode == "mpi":
        # Use dask-mpi for initialization
        from dask_mpi import initialize

        logger.info("Initializing Dask with MPI")
        initialize(interface="bond0")
        client = Client()
    else:
        # Manual mode: launch scheduler and workers with proper configuration
        logger.info("Launching Dask scheduler and workers manually")

        # Get worker configuration
        config_dict, config_name = get_worker_config(args)
        logger.info(f"Using {config_name}")

        # Launch scheduler
        scheduler_proc = launch_scheduler(interface="bond0", logger=logger)

        # Connect to scheduler
        scheduler_file = get_scheduler_file()

        client = Client(scheduler_file=scheduler_file, timeout="60s")
        if args.processpool and args.one_worker_per_node:
            logger.info("Registering processpool as 'processes' executor")
            client.register_plugin(AddProcessPool())
        logger.info("Connected to scheduler")

        # Launch workers with configuration
        worker_procs = launch_workers_with_config(
            config_dict,
            interface="bond0",
            logger=logger,
            one_worker_per_node=args.one_worker_per_node,
        )

    logger.info(f"Dask client: {client}")
    logger.info(f"Dashboard link: {client.dashboard_link}")

    return client, scheduler_proc, worker_procs, config_dict


def wait_for_workers_to_connect(
    client, args, config_dict, worker_procs, scheduler_proc
):
    """Wait for all workers to connect to the scheduler

    Returns:
        bool: True if all workers connected successfully, False otherwise
    """
    # Calculate expected number of workers
    nodefile = os.getenv("PBS_NODEFILE")
    n_nodes = len(open(nodefile).readlines())

    if args.scheduler_mode == "mpi":
        expected_workers = n_nodes
    else:
        # Calculate total workers from config based on mode
        if args.one_worker_per_node:
            expected_workers = n_nodes
        else:
            expected_workers = (
                sum(worker_opts["nworkers"] for worker_opts in config_dict.values())
                * n_nodes
            )

    # Wait for all workers to connect
    logger.info(f"Waiting for {expected_workers} workers to connect...")

    # Use Dask's built-in wait_for_workers
    initial_workers = len(client.scheduler_info(n_workers=-1)["workers"])
    logger.info(f"Initial workers connected: {initial_workers}")

    try:
        client.wait_for_workers(n_workers=expected_workers, timeout=600)
        logger.info("wait_for_workers completed successfully")
    except TimeoutError as e:
        current_workers = len(client.scheduler_info(n_workers=-1)["workers"])
        logger.warning(
            f"TimeoutError: Only {current_workers}/{expected_workers} workers connected: {e}"
        )
        cleanup_workers(worker_procs, scheduler_proc)
        return False
    except Exception as e:
        current_workers = len(client.scheduler_info(n_workers=-1)["workers"])
        logger.warning(
            f"Exception during wait_for_workers: {e} (type: {type(e).__name__})"
        )
        logger.warning(f"Current workers: {current_workers}/{expected_workers}")
        cleanup_workers(worker_procs, scheduler_proc)
        return False

    # Get worker info and verify count
    worker_info = client.scheduler_info(n_workers=-1)["workers"]
    actual_workers = len(worker_info)
    logger.info(f"Number of workers connected: {actual_workers}/{expected_workers}")

    if actual_workers != expected_workers:
        logger.error(
            f"Worker count mismatch! Expected {expected_workers}, got {actual_workers}"
        )
        logger.info("Connected worker details:")
        for worker_id, worker_data in worker_info.items():
            logger.info(
                f"  Worker {worker_id}: {worker_data.get('host', 'unknown')} - resources: {worker_data.get('resources', {})}"
            )

        logger.info("\nCheck logs/worker_*_stderr.log for worker startup errors")
        cleanup_workers(worker_procs, scheduler_proc)
        return False

    logger.info(f"All {expected_workers} workers connected successfully")
    return True


def execute_tasks(client, args):
    """Execute tasks based on the concurrent workers configuration"""
    # Get the number of nodes
    nodefile = os.getenv("PBS_NODEFILE")
    n_nodes = len(open(nodefile).readlines())

    # Determine if we should use processpool
    use_processpool = args.processpool and args.one_worker_per_node

    # if args.concurrent_workers == 12:
    #     # Single tile config (12 GPU workers per node)
    #     run_tasks(client, "tile config",
    #               args.num_tasks, args.sleep_time, 12, n_nodes,
    #               worker_resources={'gpu': 1}, track_timeline=args.track_timeline,
    #               use_processpool=use_processpool, use_trigger=args.trigger)

    # elif args.concurrent_workers == 24:
    #     # Two CCs config (24 workers per node)
    #     run_tasks(client, "2 ccs config",
    #               args.num_tasks, args.sleep_time, 24, n_nodes,
    #               worker_resources={'gpu': 1}, track_timeline=args.track_timeline,
    #               use_processpool=use_processpool, use_trigger=args.trigger)

    # elif args.concurrent_workers == 48:
    #     # Four CCs config (48 workers per node)
    #     run_tasks(client, "4 ccs config",
    #               args.num_tasks, args.sleep_time, 48, n_nodes,
    #               worker_resources={'gpu': 1}, track_timeline=args.track_timeline,
    #               use_processpool=use_processpool, use_trigger=args.trigger)

    # elif args.concurrent_workers == 102:
    #     # Full node config (102 CPU + 12 GPU threads per node)
    #     n_tasks = args.num_tasks

    #     # Create registry if tracking timeline
    #     registry = TaskRegistry() if args.track_timeline else None
    #     if registry:
    #         registry.set_start_time()

    #     submission_start = time.perf_counter()

    #     cpu_result = run_tasks_no_wait(
    #         client, n_tasks, args.sleep_time,
    #         worker_resources={'cpu': 1}, registry=registry, use_processpool=use_processpool, use_trigger=args.trigger
    #     )

    #     gpu_result = run_tasks_no_wait(
    #         client, n_tasks, args.sleep_time,
    #         worker_resources={'gpu': 1}, registry=registry, use_processpool=use_processpool, use_trigger=args.trigger
    #     )

    #     # Handle trigger mode vs regular mode
    #     if args.trigger:
    #         trigger_file_cpu, cpu_futures = cpu_result
    #         trigger_file_gpu, gpu_futures = gpu_result
    #     else:
    #         cpu_futures = cpu_result
    #         gpu_futures = gpu_result

    #     submission_time = time.perf_counter() - submission_start
    #     logger.info(f"Task submission completed in {submission_time:.2f} seconds")

    #     # If using trigger mode, write trigger files to start execution
    #     if args.trigger:
    #         trigger_timestamp = time.time()
    #         logger.info(f"Writing trigger files at timestamp {trigger_timestamp:.9f}...")
    #         temp_file_cpu = trigger_file_cpu + '.tmp'
    #         with open(temp_file_cpu, 'w') as f:
    #             f.write(f"{trigger_timestamp}\n")
    #         os.rename(temp_file_cpu, trigger_file_cpu)  # Atomic rename
    #         temp_file_gpu = trigger_file_gpu + '.tmp'
    #         with open(temp_file_gpu, 'w') as f:
    #             f.write(f"{trigger_timestamp}\n")
    #         os.rename(temp_file_gpu, trigger_file_gpu)  # Atomic rename
    #         logger.info("Trigger files written - tasks will now execute")

    #     # Wait for all tasks to finish with timeline tracking
    #     import math
    #     n_concurrent_slots_total = (102 + 12) * n_nodes
    #     zero_latency_runtime = args.sleep_time * math.ceil((2 * n_tasks) / n_concurrent_slots_total)
    #     n_finished, runtime = wait_for_tasks_with_timeline(
    #         cpu_futures + gpu_futures, zero_latency_runtime, registry=registry
    #     )

    #     # Write timeline to CSV if tracking
    #     if registry:
    #         timeline_file = 'logs/timeline_client.csv'
    #         registry.write_to_csv(timeline_file)
    #         logger.info(f"Timeline written to {timeline_file}")

    #     total_tasks = 2 * n_tasks
    #     if n_finished == total_tasks:
    #         logger.info(f"All CPU+GPU tasks finished in {runtime:.2f} seconds")
    #         logger.info(f"CPU+GPU Tasks per second per node: {total_tasks/runtime/n_nodes:.2f}")
    #         logger.info(f"CPU+GPU Runtime - Zero latency runtime: {runtime - zero_latency_runtime:.2f} seconds")
    #     else:
    #         logger.warning(f"Not all CPU+GPU tasks finished, only {n_finished} out of {total_tasks} tasks")
    #         logger.info(f"CPU+GPU Runtime is {runtime:.2f} seconds")

    # else:
    # Generic config for any number of workers
    run_tasks(
        client,
        f"{args.concurrent_workers} workers config",
        args.num_tasks,
        args.sleep_time,
        args.concurrent_workers,
        n_nodes,
        track_timeline=args.track_timeline,
        use_processpool=use_processpool,
        use_trigger=args.trigger,
    )


def main():
    args = arg_parse()
    for arg_name, arg_value in vars(args).items():
        logger.info(f"Argument '{arg_name}': {arg_value}")

    scheduler_proc = None
    worker_procs = []

    # Setup signal handler for cleanup
    def signal_handler(sig, frame):
        cleanup_workers(worker_procs, scheduler_proc)
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Initialize Dask client
        client, scheduler_proc, worker_procs, config_dict = initialize_dask_client(args)

        # Wait for workers to connect
        if not wait_for_workers_to_connect(
            client, args, config_dict, worker_procs, scheduler_proc
        ):
            sys.exit(1)

        # Execute tasks
        execute_tasks(client, args)

        logger.info("Goodbye Dask")
        client.shutdown()
    finally:
        cleanup_workers(worker_procs, scheduler_proc)


if __name__ == "__main__":
    main()
