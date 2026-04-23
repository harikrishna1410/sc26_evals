import argparse
import os
from glob import glob

import pandas as pd
from mpi4py import MPI


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy", type=str, default="unknown", help="Policy name"
    )
    args = parser.parse_args()
    policy = args.policy

    comm = MPI.COMM_WORLD
    size = comm.size
    rank = comm.rank

    # Get node-local files (each node has its own /tmp)
    fnames = sorted(glob(f"/tmp/mpi_timeline_{policy}_*.csv"))
    out_name = f"mpi_timeline_worker_{policy}"

    with open(os.environ.get("PBS_NODEFILE", "/dev/null"), "r") as f:
        nodes = f.readlines()

    nnodes = len(nodes) if len(nodes) > 0 else 1

    ppn = size // nnodes

    local_rank = rank % ppn

    # Distribute node-local files among local ranks on this node
    files_per_rank = (len(fnames) + ppn - 1) // ppn  # Ceiling division

    start_idx = files_per_rank * local_rank
    end_idx = min(files_per_rank * (local_rank + 1), len(fnames))
    # Distribute files across ranks
    my_fnames = fnames[start_idx:end_idx]
    # Read and merge local files
    local_dfs = []
    for fname in my_fnames:
        try:
            df = pd.read_csv(fname)
            local_dfs.append(df)
        except Exception as e:
            print(f"Rank {rank}: Error reading {fname}: {e}")

    # Merge local dataframes
    if local_dfs:
        local_merged = pd.concat(local_dfs, ignore_index=True)
    else:
        local_merged = pd.DataFrame(
            columns=["task_id", "policy", "nnodes", "start_time(s)", "end_time(s)", "elapsed_time(s)", "sleep_time(s)"]
        )

    # Gather all dataframes to rank 0
    all_dfs = comm.gather(local_merged, root=0)

    # Rank 0 merges everything and saves
    if rank == 0:
        final_df = pd.concat(all_dfs, ignore_index=True)
        final_df = final_df.sort_values(by=["start_time(s)"])
        final_df.to_csv(f"logs/{out_name}.csv", index=False)
        print(f"Merged {len(fnames)} files into {out_name}.csv")
        print(f"Total rows: {len(final_df)}")


if __name__ == "__main__":
    main()
