import argparse
import time

from mpi4py import MPI


def main(
    task_id: int,
    sleep_time: float,
    policy: str,
    nnodes: int,
):
    comm = MPI.COMM_WORLD

    rank = comm.rank

    timestamps = {}

    comm.barrier()
    timestamps["start"] = time.time()

    # Use time.perf_counter() for better precision
    start = time.perf_counter()
    while time.perf_counter() - start < sleep_time:
        pass  # Busy wait

    comm.barrier()
    timestamps["end"] = time.time()

    if rank == 0:
        with open(f"/tmp/mpi_timeline_{policy}_{task_id}.csv", "w") as f:
            f.write("task_id,policy,nnodes,start_time(s),end_time(s),elapsed_time(s),sleep_time(s)\n")
            f.write(
                f"{task_id},{policy},{nnodes},{timestamps['start']:.9f},{timestamps['end']:.9f},{timestamps['end'] - timestamps['start']},{sleep_time}\n"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_id", type=int, required=True, help="Task ID")
    parser.add_argument(
        "--sleep_time", type=float, default=0.0, help="Sleep time in seconds"
    )
    parser.add_argument(
        "--policy", type=str, default="unknown", help="Policy name"
    )
    parser.add_argument(
        "--nnodes", type=int, default=1, help="Number of nodes for this task"
    )
    args = parser.parse_args()
    main(args.task_id, args.sleep_time, args.policy, args.nnodes)
