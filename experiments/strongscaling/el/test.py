import argparse
import json
import logging
import os
import sys
import uuid

from ensemble_launcher import EnsembleLauncher, write_results_to_json
from ensemble_launcher.config import LauncherConfig, PolicyConfig, SystemConfig
from ensemble_launcher.ensemble import Task
from ensemble_launcher.helper_functions import get_nodes
from utils import sleep_task

CHECKPOINT_DIR = os.path.join("/tmp", f".ckpt_{str(uuid.uuid4())}")
os.makedirs(CHECKPOINT_DIR)

##configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Run ensemble launcher with specified parameters"
    )
    parser.add_argument(
        "--sleep-time", type=float, default=10.0, help="sleep time for tasks in seconds"
    )
    parser.add_argument(
        "--num-tasks", type=int, default=10, help="number of tasks per worker"
    )
    parser.add_argument(
        "--concurrent-workers", type=int, default=64, help="number of tasks per worker"
    )
    parser.add_argument(
        "--nlevels", type=int, default=2, help="number of levels in the launcher"
    )

    args = parser.parse_args()

    sleep_time = args.sleep_time

    # Calculate total number of tasks
    nodes = get_nodes()
    ntasks = args.num_tasks

    # Create tasks directly
    tasks = {}
    for i in range(ntasks):
        tasks[f"task_{i}"] = Task(
            task_id=f"task_{i}",
            nnodes=1,
            ppn=1,
            executable=sleep_task,
            args=(i, sleep_time),
        )

    ##create the system config
    cpus = list(range(104))
    cpus.pop(52)  # can't use these cores on Aurora
    cpus.pop(0)  # can't use these cores on Aurora
    gpus = [i for i in range(12)]

    sys_config = SystemConfig(
        name="Aurora", cpus=cpus[: args.concurrent_workers], gpus=gpus
    )

    ##set some environment variables
    os.environ["ZE_FLAT_DEVICE_HIERARCHY"] = "FLAT"

    ##create the launcher config
    launcher_config = LauncherConfig(
        comm_name="async_zmq",
        child_executor_name="async_mpi",
        task_executor_name="async_processpool",
        worker_logs=False,
        master_logs=True,
        log_level=logging.INFO,
        report_interval=10.0,
        children_scheduler_policy="fixed_leafs_children_policy",
        policy_config=PolicyConfig(nlevels=args.nlevels, leaf_nodes=len(nodes)),
        checkpoint_dir=CHECKPOINT_DIR,
        heartbeat_interval=5.0,
        result_flush_interval = 0.1,
    )
    ##create the EnsembleLauncher class
    el = EnsembleLauncher(
        ensemble_file=tasks,
        system_config=sys_config,
        launcher_config=launcher_config,
        async_orchestrator=True,
    )

    results = el.run()  ##blocking call
    results_dict = results.to_dict()
    with open("logs/results.json", "w") as f:
        json.dump(results_dict, f, indent=4)
