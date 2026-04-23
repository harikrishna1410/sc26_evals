import argparse
import logging
import os
import threading
import time
from concurrent.futures import Future, TimeoutError

from config_injob import (
    aurora_four_ccs_config,
    aurora_single_core_tile_config,
    aurora_single_tile_config,
    aurora_two_ccs_config,
    get_nworker_config,
)
from utils import sleep_task

import parsl
from parsl import python_app


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



run_serial_example_cpu = python_app(sleep_task,executors=["cpu"])
run_serial_example_gpu = python_app(sleep_task,executors=["gpu"])


def arg_parse():
    parser = argparse.ArgumentParser()    
    parser.add_argument('--sleep-time', type=float, default=10.0,
                        help='sleep time for tasks in seconds')
    parser.add_argument('--num-tasks', type=int, default=10,
                        help='total number of tasks to run')
    parser.add_argument('--concurrent-workers', type=int, default=12,
                        help='number of concurrent workers')
    parser.add_argument('--track-timeline', action='store_true',
                        help='track and log detailed task timeline')
    parser.add_argument('--trigger', action='store_true',
                        help='Use Future-based trigger to coordinate task execution (default: immediate execution)')
    
    return parser.parse_args()


def wait_for_tasks(futures, task_type, zero_latency_runtime):
    """Wait for all tasks to finish and return completion statistics"""
    n_tasks = len(futures)
    logging.info(f"Waiting for {n_tasks} {task_type} tasks to finish")
    
    start_time = time.perf_counter()
    n_finished = 0
    
    for i, f in enumerate(futures):
        try:
            timeout = zero_latency_runtime * 10 + 300.0
            f.result(timeout=timeout)
            n_finished += 1
        except TimeoutError:
            logging.warning(f"{task_type} Task {i} timed out")
    
    end_time = time.perf_counter()
    runtime = end_time - start_time
    
    return n_finished, runtime



def run_tasks_no_wait(app_func, task_type, n_tasks, sleep_time, registry=None, trigger=None):
    """Submit tasks without waiting for completion
    
    Args:
        n_tasks: Total number of tasks to run
        registry: Optional TaskRegistry for timeline tracking
        trigger: Optional Future to coordinate task execution start
    """
    
    logging.info(f"Running {n_tasks} {task_type} tasks")
    
    # Create a list of futures
    futures = []
    for i in range(n_tasks):
        if registry:
            registry.register_submission(i)
        
        # Pass trigger as dependency if provided
        if trigger:
            f = app_func(task_id=i, sleep_time=sleep_time, trigger=trigger)
        else:
            f = app_func(task_id=i, sleep_time=sleep_time)
        
        # Add done callback if tracking timeline
        if registry:
            f.add_done_callback(lambda fut, tid=i: registry.register_completion(tid))
        
        futures.append(f)
    
    return futures


def run_tasks(app_func, config_name, task_type, n_tasks, sleep_time, 
              n_concurrent_slots_per_node, n_nodes, track_timeline=False, use_trigger=False):
    """Run tasks with the given configuration and parameters
    
    Args:
        n_tasks: Total number of tasks to run
        n_concurrent_slots_per_node: Number of concurrent task slots per node
        n_nodes: Number of nodes
        track_timeline: Whether to track and log detailed task timeline
        use_trigger: Whether to use Future-based trigger for coordinated execution
    """
    
    logging.info(f"Hello Parsl ({config_name})")
    
    import math
    n_concurrent_slots_total = n_concurrent_slots_per_node * n_nodes
    zero_latency_runtime = sleep_time * math.ceil(n_tasks / n_concurrent_slots_total)
    
    # Warmup: submit and wait for a single task to complete
    logging.info("Warming up Parsl with 10 tasks...")
    for _ in range(10):
        warmup_future = app_func(task_id=-1, sleep_time=0.1)
        warmup_future.result()
    logging.info("Warmup complete")
    
    # Create registry if tracking timeline
    registry = TaskRegistry() if track_timeline else None
    if registry:
        registry.set_start_time()
    
    # Create a trigger Future to coordinate task execution if requested
    trigger = Future() if use_trigger else None
    if use_trigger:
        logging.info("Using trigger mode - tasks will wait for trigger before execution")
    
    # Record submission start time
    submission_start = time.perf_counter()
    
    futures = run_tasks_no_wait(app_func, task_type, n_tasks, sleep_time, registry=registry, trigger=trigger)
    
    submission_time = time.perf_counter() - submission_start
    logging.info(f"Task submission completed in {submission_time:.2f} seconds")
    
    # Now trigger all tasks to start by resolving the trigger Future
    if use_trigger:
        logging.info("Triggering task execution...")
        trigger.set_result(True)
    
    # Wait for all tasks to finish
    n_finished, runtime = wait_for_tasks(futures, task_type, zero_latency_runtime)
    
    # Write timeline to CSV if tracking
    if registry:
        timeline_file = f'logs/timeline_{task_type}.csv'
        registry.write_to_csv(timeline_file)
        logging.info(f"Timeline written to {timeline_file}")
    
    if n_finished == n_tasks:
        logging.info(f"All {task_type} tasks finished in {runtime:.2f} seconds")
        logging.info(f"{task_type} Tasks per second per node: {n_tasks/runtime/n_nodes:.2f}")
        logging.info(f"{task_type} Runtime - Zero latency runtime: {runtime - zero_latency_runtime:.2f} seconds")
    else:
        logging.warning(f"Not all {task_type} tasks finished, only {n_finished} out of {n_tasks} tasks")
        logging.info(f"{task_type} Runtime is {runtime:.2f} seconds")


# Main function
if __name__ == "__main__":

    args = arg_parse()
    
    # Set up logging to file
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/main.log'),
            logging.StreamHandler()
        ]
    )
    
    for arg_name, arg_value in vars(args).items():
        logging.info(f"Argument '{arg_name}': {arg_value}")
    
    # Get the number of nodes from the PBS_NODEFILE environment variable
    nodefile = os.getenv("PBS_NODEFILE")
    n_nodes = len(open(nodefile).readlines())

    # if args.concurrent_workers == 12:
    #     with parsl.load(aurora_single_tile_config):
    #         run_tasks(run_serial_example_gpu, "tile config", "GPU", 
    #               args.num_tasks, args.sleep_time, 12, n_nodes, track_timeline=args.track_timeline, use_trigger=args.trigger)
    # elif args.concurrent_workers == 24:
    #     with parsl.load(aurora_two_ccs_config):
    #         run_tasks(run_serial_example_gpu, "2 ccs config", "GPU", 
    #               args.num_tasks, args.sleep_time, 24, n_nodes, track_timeline=args.track_timeline, use_trigger=args.trigger)
    # elif args.concurrent_workers == 48:
    #     with parsl.load(aurora_four_ccs_config):
    #         run_tasks(run_serial_example_gpu, "4 ccs config", "GPU", 
    #               args.num_tasks, args.sleep_time, 48, n_nodes, track_timeline=args.track_timeline, use_trigger=args.trigger)
    # elif args.concurrent_workers == 102:
    #     with parsl.load(aurora_single_core_tile_config):
    #         n_tasks = args.num_tasks
            
    #         # Create registry if tracking timeline
    #         registry = TaskRegistry() if args.track_timeline else None
    #         if registry:
    #             registry.set_start_time()
            
    #         # Create a trigger Future to coordinate task execution if requested
    #         trigger = Future() if args.trigger else None
    #         if args.trigger:
    #             logging.info("Using trigger mode - tasks will wait for trigger before execution")
            
    #         submission_start = time.perf_counter()
            
    #         cpu_futures = run_tasks_no_wait(run_serial_example_cpu, "CPU", n_tasks, args.sleep_time, registry=registry, trigger=trigger)
    #         gpu_futures = run_tasks_no_wait(run_serial_example_gpu, "GPU", n_tasks, args.sleep_time, registry=registry, trigger=trigger)
            
    #         submission_time = time.perf_counter() - submission_start
    #         logging.info(f"Task submission completed in {submission_time:.2f} seconds")
            
    #         # Now trigger all tasks to start by resolving the trigger Future
    #         if args.trigger:
    #             logging.info("Triggering task execution...")
    #             trigger.set_result(True)
            
    #         # Wait for all tasks to finish
    #         import math
    #         n_concurrent_slots_total = (102 + 12) * n_nodes
    #         zero_latency_runtime = args.sleep_time * math.ceil((2 * n_tasks) / n_concurrent_slots_total)
    #         n_finished, runtime = wait_for_tasks(cpu_futures + gpu_futures, "CPU+GPU", zero_latency_runtime)
            
    #         # Write timeline to CSV if tracking
    #         if registry:
    #             timeline_file = 'logs/timeline_CPU+GPU.csv'
    #             registry.write_to_csv(timeline_file)
    #             logging.info(f"Timeline written to {timeline_file}")
            
    #         total_tasks = 2 * n_tasks
    #         if n_finished == total_tasks:
    #             logging.info(f"All CPU+GPU tasks finished in {runtime:.2f} seconds")
    #             logging.info(f"CPU+GPU Tasks per second per node: {total_tasks/runtime/n_nodes:.2f}")
    #             logging.info(f"CPU+GPU Runtime - Zero latency runtime: {runtime - zero_latency_runtime:.2f} seconds")
    #         else:
    #             logging.warning(f"Not all CPU+GPU tasks finished, only {n_finished} out of {total_tasks} tasks")
    #             logging.info(f"CPU+GPU Runtime is {runtime:.2f} seconds")
    # else:
    config = get_nworker_config(nworkers=args.concurrent_workers)
    with parsl.load(config):
        run_tasks(run_serial_example_cpu, "cpu config", "CPU", 
                args.num_tasks, args.sleep_time, args.concurrent_workers, n_nodes, track_timeline=args.track_timeline, use_trigger=args.trigger)    
    
    logging.info("Goodbye Parsl")
