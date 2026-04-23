def sleep_task(task_id:str, sleep_time:float):
    import time

    timestamps = {}
    timestamps["start"] = time.time()
    
    # Use time.perf_counter() for better precision
    start = time.perf_counter()
    while time.perf_counter() - start < sleep_time:
        pass  # Busy wait
    
    timestamps["end"] = time.time()
    with open(f"/tmp/timeline_{task_id}.csv","w") as f:
        f.write("task_id,start_time(s),end_time(s),elapsed_time(s),sleep_time(s)\n")
        f.write(f"{task_id},{timestamps['start']:.9f},{timestamps['end']:.9f},{timestamps['end']-timestamps['start']},{sleep_time}\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_id", type=str, required=True, help="Task ID")
    parser.add_argument("--sleep_time", type=float, default=0.0, help="Sleep time in seconds")
    args = parser.parse_args()
    sleep_task(args.task_id, args.sleep_time)