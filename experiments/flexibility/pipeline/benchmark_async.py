import argparse
import csv
import json
import multiprocessing
import os
import queue

os.environ["EL_EXTERNAL_POLICY_PATH"] = os.path.join(os.getcwd(), "policies")
os.environ["EL_EXTERNAL_POLICY_MODULE"] = "external_policies"
import time
import uuid
from dataclasses import dataclass
from typing import Dict

from ensemble_launcher import EnsembleLauncher
from ensemble_launcher.config import (
    LauncherConfig,
    PolicyConfig,
    SystemConfig,
)
from ensemble_launcher.ensemble import Task
from ensemble_launcher.helper_functions import get_nodes
from ensemble_launcher.logging import setup_logger
from ensemble_launcher.orchestrator import ClusterClient

cwd = os.getcwd()
CKPT_DIR = os.path.join(cwd, f"ckpt_{str(uuid.uuid4())}")
os.makedirs(CKPT_DIR, exist_ok=True)


def load_config(path):
    with open(path) as f:
        return json.load(f)


class PipelineRegistry:
    """Tracks wall-clock start/end timestamps per task, keyed by task_id."""

    def __init__(self):
        # {task_id: {"stage": str, "start": float, "end": float}}
        self._data: Dict[str, Dict] = {}

    def record(self, task_id, stage, event, timestamp):
        if task_id not in self._data:
            self._data[task_id] = {
                "stage": stage,
                "start": 0.0,
                "end": 0.0,
            }
        if event == "start":
            self._data[task_id]["start"] = timestamp
        elif event == "end":
            self._data[task_id]["end"] = timestamp

    def to_csv(self, path: str):
        header = ["task_id", "stage", "start", "end"]
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for task_id in sorted(self._data):
                entry = self._data[task_id]
                writer.writerow(
                    [
                        task_id,
                        entry["stage"],
                        entry["start"],
                        entry["end"],
                    ]
                )
        return path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "default_config.json"),
        help="Path to JSON config defining pipeline stages",
    )
    parser.add_argument(
        "--npipelines",
        type=int,
        default=4,
        help="Number of pipelines run in parallel",
    )
    parser.add_argument(
        "--policy", type=str, default="fixed_leafs_children_policy", help="policy name"
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help="Directory for log files and pipeline timestamps CSV",
    )
    parser.add_argument(
        "--use_tags",
        action="store_true",
        help="use tag to get policy config",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Generic task submission — driven entirely by stage config
# ---------------------------------------------------------------------------


def submit_task(client, stage_cfg, task_index):
    """Submit a single task based on stage config.

    Task type is derived from total processes (np) and total GPUs:
      - np > 1 (MPI)          -> async_mpi executor, mpi_example.py, tag="sim"
      - np == 1, GPUs > 0     -> async_mpi_processpool, sleep_task,   tag="inference"
      - np == 1, GPUs == 0    -> async_mpi_processpool, sleep_task,   tag="post"
    """
    name = stage_cfg["name"]
    task_id = f"{name}-{task_index}"
    nnodes = stage_cfg["nnodes"]
    ppn = stage_cfg["ppn"]
    ngpus = stage_cfg.get("ngpus_per_process", 0)
    duration = stage_cfg["duration_sec"]

    np = nnodes * ppn
    total_gpus = ngpus * np

    # Use executor from config if provided, otherwise derive from np
    executor = stage_cfg.get(
        "executor_name", "async_mpi" if np > 1 else "async_mpi_processpool"
    )

    if np > 1:
        # MPI task
        tag = "sim"
        tasks = [
            Task(
                task_id=task_id,
                nnodes=nnodes,
                ppn=ppn,
                ngpus_per_process=ngpus,
                executable=(
                    f"python3 {cwd}/mpi_example.py"
                    f" --duration {duration}"
                    f" --task_id {task_id}"
                    f" --ncpus {np}"
                    f" --ngpus {total_gpus}"
                    f" --task_type {name}"
                ),
                executor_name=executor,
                tag=tag,
            )
        ]
    else:
        # Serial task
        from serial_example import sleep_task

        tag = "inference" if total_gpus > 0 else "post"
        tasks = [
            Task(
                task_id=task_id,
                nnodes=nnodes,
                ppn=ppn,
                ngpus_per_process=ngpus,
                executable=sleep_task,
                args=(task_id, duration, np, total_gpus, name),
                executor_name=executor,
                tag=tag,
            )
        ]

    return client.submit_batch(tasks=tasks)


# ---------------------------------------------------------------------------
# Stage-process architecture: each stage is its own process with queues
# ---------------------------------------------------------------------------


@dataclass
class WorkItem:
    task_index: int = 0


def stage_process(stage_cfg, in_q, out_q, timing_q, log_dir, ckpt_dir):
    """Generic stage worker: reads WorkItems from in_q, submits tasks, pushes results to out_q.

    Fanout handling:
      - fanout >= 1: each completed task emits int(fanout) items to out_q
      - fanout < 1:  accumulate int(1/fanout) completions, then emit 1 item
    """
    stage_name = stage_cfg["name"]
    next_fanout = stage_cfg.get("fanout", 1.0)

    logger = setup_logger(name=f"stage-{stage_name}", log_dir=log_dir)
    logger.info(f"Stage process '{stage_name}' started (fanout={next_fanout})")

    client = ClusterClient(checkpoint_dir=ckpt_dir)
    client.start()

    pending = {}  # future -> WorkItem
    got_sentinel = False

    # Fanout bookkeeping
    if next_fanout >= 1:
        emit_per_completion = int(next_fanout)
        accumulate_threshold = 1
    else:
        emit_per_completion = 0  # signals accumulation mode
        accumulate_threshold = int(1.0 / next_fanout)

    # Number of completions accumulated — used when fanout < 1
    completion_count = 0
    # Counter for generating unique task_index values in out_q
    out_task_counter = 0

    def _handle_item(item):
        task_id = f"{stage_name}-{item.task_index}"
        timing_q.put((task_id, stage_name, "start", time.time()))
        futures = submit_task(client, stage_cfg, item.task_index)
        logger.info(f"Stage '{stage_name}': submitted task {task_id}")
        for f in futures:
            pending[f] = item

    def _emit(count):
        nonlocal out_task_counter
        for _ in range(count):
            out_q.put(WorkItem(task_index=out_task_counter))
            out_task_counter += 1

    try:
        while True:
            # Drain input queue (non-blocking) to pick up new work
            while True:
                try:
                    item = in_q.get_nowait()
                except queue.Empty:
                    break
                if item is None:
                    got_sentinel = True
                    logger.info(f"Stage '{stage_name}' received shutdown sentinel")
                    continue
                _handle_item(item)

            # If nothing pending and no sentinel yet, block briefly on queue
            if not pending and not got_sentinel:
                try:
                    item = in_q.get(timeout=0.1)
                    if item is None:
                        got_sentinel = True
                        logger.info(f"Stage '{stage_name}' received shutdown sentinel")
                    else:
                        _handle_item(item)
                except queue.Empty:
                    pass
                continue

            # Check completed futures
            done_futures = [f for f in pending if f.done()]
            for f in done_futures:
                item = pending.pop(f)
                exc = f.exception()
                task_id = f"{stage_name}-{item.task_index}"
                if exc is not None:
                    logger.error(
                        f"Stage '{stage_name}': task {task_id} failed: {exc}",
                        exc_info=exc,
                    )
                    timing_q.put((task_id, stage_name, "end", time.time()))
                    continue

                timing_q.put((task_id, stage_name, "end", time.time()))
                logger.info(f"Stage '{stage_name}': task {task_id} completed")

                # Fan out to next stage
                if emit_per_completion > 0:
                    _emit(emit_per_completion)
                else:
                    completion_count += 1
                    if completion_count >= accumulate_threshold:
                        _emit(1)
                        completion_count = 0

            # Exit when sentinel received and all in-flight work is done
            if got_sentinel and not pending:
                # Flush any remaining accumulated completions
                if completion_count > 0:
                    _emit(1)
                out_q.put(None)  # propagate sentinel to next stage
                logger.info(f"Stage '{stage_name}' shutting down")
                break

            time.sleep(0.01)  # avoid busy-spin
    finally:
        client.teardown()


def run_stage_pipeline(config, args, logger, ckpt_dir):
    """Drive all pipelines using one process per stage connected by queues."""
    stage_configs = config["stages"]
    nstages = len(stage_configs)

    # Create nstages + 1 queues (input for each stage + final done_q)
    queues = [multiprocessing.Queue() for _ in range(nstages + 1)]
    timing_q = multiprocessing.Queue()

    # Spawn stage processes
    processes = []
    for i, stage_cfg in enumerate(stage_configs):
        # Last stage gets fanout=1.0 (items land in done_q as-is)
        if i == nstages - 1:
            stage_cfg = {**stage_cfg, "fanout": 1.0}
        p = multiprocessing.Process(
            target=stage_process,
            args=(
                stage_cfg,
                queues[i],
                queues[i + 1],
                timing_q,
                args.log_dir,
                ckpt_dir,
            ),
            name=f"stage-{stage_cfg['name']}",
        )
        p.start()
        processes.append(p)
        logger.info(f"Spawned stage process: {stage_cfg['name']} (pid={p.pid})")

    # Seed first queue with work items
    for idx in range(args.npipelines):
        queues[0].put(WorkItem(task_index=idx))
    queues[0].put(None)  # sentinel

    # Compute expected completions from fanout chain
    fanouts = [s.get("fanout", 1.0) for s in stage_configs]
    total_fanout = 1.0
    for f in fanouts[:-1]:  # last stage fanout doesn't multiply
        total_fanout *= f
    total_expected = max(1, int(args.npipelines * total_fanout))
    logger.info(f"Waiting for ~{total_expected} items in done_q")

    # Drain done_q
    done_q = queues[-1]
    completed = 0
    while True:
        item = done_q.get()
        if item is None:
            logger.info("Received sentinel from last stage — all work done")
            break
        completed += 1
        logger.info(
            f"Task task_index={item.task_index} "
            f"completed ({completed}/{total_expected})"
        )

    # Join all stage processes
    for p in processes:
        p.join(timeout=30)
        if p.is_alive():
            logger.warning(f"Stage process {p.name} did not exit, terminating")
            p.terminate()

    # Drain timing_q and build registry
    registry = PipelineRegistry()
    while True:
        try:
            task_id, stage_name, event, timestamp = timing_q.get_nowait()
        except queue.Empty:
            break
        registry.record(task_id, stage_name, event, timestamp)

    csv_path = registry.to_csv(
        os.path.join(cwd, args.log_dir, "pipeline_timestamps.csv")
    )
    logger.info(f"Pipeline timestamps written to {csv_path}")
    logger.info(f"All {completed} tasks completed across {args.npipelines} pipelines")


def get_policy_config(config, args, logger):
    stage_configs = config["stages"]
    stage_by_name = {s["name"]: s for s in stage_configs}

    nnodes = len(get_nodes())
    if args.use_tags:
        inference_gpu_sec = 0.0
        sim_gpu_sec = 0.0
        post_gpu_sec = 0.0
        ntasks = args.npipelines * 1.0
        max_nnodes = -1
        nodes_per_sim = -1
        for s in stage_configs:
            if s["tag"] == "inference":
                inference_gpu_sec += (
                    ntasks
                    * s["nnodes"]
                    * s["ppn"]
                    * s["ngpus_per_process"]
                    * s["duration_sec"]
                )
            elif s["tag"] == "sim":
                sim_gpu_sec += (
                    ntasks
                    * s["nnodes"]
                    * s["ppn"]
                    * s["ngpus_per_process"]
                    * s["duration_sec"]
                )
                nodes_per_sim = max(nodes_per_sim, s["nnodes"])
            elif s["tag"] == "post":
                post_gpu_sec += (
                    ntasks
                    * s["nnodes"]
                    * s["ppn"]
                    * s.get("ngpus_per_process", 0)
                    * s["duration_sec"]
                )
            ntasks *= s["fanout"]
            max_nnodes = max(max_nnodes, s["nnodes"])
        sim_nodes = max(
            int(sim_gpu_sec * nnodes / (inference_gpu_sec + sim_gpu_sec)), nodes_per_sim
        )
        logger.info(
            f"Using tags got inference gpu sec: {inference_gpu_sec}, sim gpu sec: {sim_gpu_sec}, sim_nodes: {sim_nodes}"
        )
        return PolicyConfig(
            nlevels=2,
            nchildren=int(nnodes // max_nnodes),
            leaf_nodes=int(nnodes // max_nnodes),
            sim_nodes=sim_nodes,
            nodes_per_sim=2,
        )
    else:
        inference_duration_sec = stage_by_name["inference"]["duration_sec"]
        inference_fanout = stage_by_name["inference"]["fanout"]
        post_fanout = stage_by_name["post"]["fanout"]
        sim_duration_sec = stage_by_name["sim"]["duration_sec"]
        nodes_per_sim = stage_by_name["sim"]["nnodes"]
        ##
        inference_gpu_secs = args.npipelines * inference_duration_sec
        ##

        sim_gpu_secs = (
            inference_fanout * post_fanout * args.npipelines * 24 * sim_duration_sec
        )

        sim_nodes = (
            int(
                int(sim_gpu_secs * nnodes / (inference_gpu_secs + sim_gpu_secs))
                // nodes_per_sim
            )
            * nodes_per_sim
        )

        # Find the largest nnodes across all stages for sim-like sizing
        max_nnodes = max(s["nnodes"] for s in stage_configs)

        return PolicyConfig(
            nlevels=2,
            nchildren=int(nnodes // max_nnodes),
            leaf_nodes=int(nnodes // max_nnodes),
            sim_nodes=sim_nodes,
            nodes_per_sim=nodes_per_sim,
        )


def main():
    args = parse_args()
    config = load_config(args.config)
    logger = setup_logger(name="script", log_dir=args.log_dir)

    logger.info(f"Starting benchmark with args: {vars(args)}")
    logger.info(f"Pipeline config: {json.dumps(config, indent=2)}")

    policy_config = get_policy_config(config, args, logger)

    # Derive executor names: use config value if provided, otherwise infer from np
    executor_names = list(
        {
            s.get(
                "executor_name",
                "async_mpi" if s["nnodes"] * s["ppn"] > 1 else "async_mpi_processpool",
            )
            for s in config["stages"]
        }
    )

    launcher_config = LauncherConfig(
        child_executor_name="async_mpi",
        task_executor_name=executor_names,
        master_logs=True,
        worker_logs=False,
        children_scheduler_policy=args.policy,
        policy_config=policy_config,
        cluster=True,
        return_stdout=True,
        checkpoint_dir=CKPT_DIR,
        overload_orchestrator_core=True,
        log_dir=args.log_dir,
        task_flush_interval=5.0,
        result_flush_interval=5.0,
        heartbeat_dead_threshold=120,
        heartbeat_interval=10.0,
    )

    cpus = list(range(104))
    cpus.pop(52)
    cpus.pop(0)
    gpus = list(range(12))

    sys_config = SystemConfig(name="aurora", ncpus=102, cpus=cpus, gpus=gpus, ngpus=12)
    logger.info(f"System config: ncpus=102, ngpus=12")

    el = EnsembleLauncher(
        ensemble_file={}, system_config=sys_config, launcher_config=launcher_config
    )
    logger.info("Starting EnsembleLauncher")
    el.start()

    time.sleep(15.0)
    logger.info(
        f"Launching {args.npipelines} pipelines using stage-process architecture"
    )
    tic = time.perf_counter()
    run_stage_pipeline(config, args, logger, CKPT_DIR)
    toc = time.perf_counter()
    logger.info(f"Executing {args.npipelines} pipelines took {toc - tic:.2f} seconds")

    el.stop()


if __name__ == "__main__":
    main()
