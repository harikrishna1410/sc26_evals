def sleep_task(task_id: str, sleep_time: float, ncpus: int = 1, ngpus: int = 0, task_type: str = "unknown", trigger=None):
    import time

    # Wait for trigger (used for coordinating task execution)
    # The trigger value is not used, just ensures dependency
    _ = trigger

    timestamps = {}
    timestamps["start"] = time.time()

    # Use time.perf_counter() for better precision
    start = time.perf_counter()
    while time.perf_counter() - start < sleep_time:
        pass  # Busy wait

    timestamps["end"] = time.time()
    with open(f"/tmp/timeline_{task_id}.csv", "w") as f:
        f.write("task_id,task_type,start_time(s),end_time(s),elapsed_time(s),sleep_time(s),ncpus,ngpus\n")
        f.write(
            f"{task_id},{task_type},{timestamps['start']:.9f},{timestamps['end']:.9f},{timestamps['end'] - timestamps['start']},{sleep_time},{ncpus},{ngpus}\n"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--task_id", type=str, required=True, help="Task ID")
    parser.add_argument(
        "--duration", type=float, default=0.0, help="Sleep time in seconds"
    )
    parser.add_argument(
        "--ncpus", type=int, default=1, help="Number of CPUs used by this task"
    )
    parser.add_argument(
        "--ngpus", type=int, default=0, help="Number of GPUs used by this task"
    )
    parser.add_argument(
        "--task_type", type=str, default="unknown", help="Type of task (e.g., inference, post, sim, training)"
    )
    args = parser.parse_args()
    sleep_task(args.task_id, args.duration, args.ncpus, args.ngpus, args.task_type)
