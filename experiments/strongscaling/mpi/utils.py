def sleep_task(task_id:str, sleep_time:float = 0.0):
    import time
    import csv

    timestamps = {}
    timestamps["start"] = time.time_ns()
    while time.time_ns() - timestamps["start"] < sleep_time*(1_000_000_000):
        pass
    timestamps["end"] = time.time_ns()
    with open(f"/tmp/timeline_{task_id}.csv","w") as f:
        writer = csv.writer(f)
        writer.writerow(["task_id", "event", "timestamp"])
        writer.writerow([task_id, "start", timestamps["start"]])
        writer.writerow([task_id, "end", timestamps["end"]])

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_id", type=str, required=True, help="Task ID")
    parser.add_argument("--sleep_time", type=float, default=0.0, help="Sleep time in seconds")
    parser.add_argument("--ntasks", type=int, default=0.0, help="Sleep time in seconds")
    args = parser.parse_args()
    for i in range(args.ntasks):
        sleep_task(int(args.task_id) * args.ntasks + i, args.sleep_time)