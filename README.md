# EnsembleLauncher SC26 Evaluation

This repository contains source files for the EnsembleLauncher paper at SC26. It includes benchmarking scripts, environment configurations, and post-processing tools for evaluating EnsembleLauncher against MPI, Parsl, and Dask across scaling, latency, throughput, and scheduling flexibility experiments.

## Repository Structure

```
sc26_evals/
├── envs/                                    # Virtual environment configurations
│   ├── el/                                  # EnsembleLauncher environment
│   │   ├── ensemble_launcher/               # EL source package
│   │   └── create_env.sh
│   ├── parsl/                               # Parsl environment
│   ├── dask/                                # Dask environment
│   └── parsl_dask/                          # Combined Parsl+Dask environment
├── experiments/
│   ├── weakscaling/                         # Weak scaling experiments
│   │   ├── el/                              # EnsembleLauncher (depth=2)
│   │   ├── el_cluster/                      # EL cluster mode
│   │   ├── el_cluster_1level_ws/            # EL depth=1 with work-stealing
│   │   ├── el_cluster_2level_ws/            # EL depth=2 with work-stealing
│   │   ├── mpi/                             # MPI baseline
│   │   ├── parsl_with_trigger/              # Parsl
│   │   ├── dask_1level_with_trigger/        # Dask depth=0
│   │   ├── dask_2level_processpool_with_trigger/  # Dask depth=1
│   │   └── post/                            # Plotting scripts
│   ├── strongscaling/                       # Strong scaling experiments
│   │   ├── el/                              # EL depth=2
│   │   ├── el_cluster/                      # EL cluster mode
│   │   ├── el_cluster_2level_ws/            # EL depth=2 with work-stealing
│   │   ├── mpi/                             # MPI baseline
│   │   ├── parsl_with_trigger/              # Parsl
│   │   ├── dask_2level_processpool_with_trigger/  # Dask depth=1
│   │   └── post/                            # Plotting scripts
│   ├── microbenchmarks/                     # Low-level performance tests
│   │   ├── roundtrip_latency/               # Per-task latency measurement
│   │   ├── worker_throughput/               # Tasks/sec throughput measurement
│   │   └── post/                            # Plotting scripts
│   └── flexibility/                         # Scheduling policy experiments
│       ├── parametric_sweep/                # Policy comparison under varying task distributions
│       ├── pipeline/                        # Multi-stage pipeline workloads
│       └── post/                            # Plotting scripts
└── README.md
```

## Environment Setup

### Prerequisites

- Python 3.10+
- MPICH (EnsembleLauncher assumes MPICH; required for `mpi4py` and MPI-based experiments)
- PBS job scheduler (for submitting experiments on an HPC cluster)

### Creating Environments

Each framework has its own virtual environment under `envs/`. Run the corresponding `create_env.sh` to set it up. All environments are created at `$HOME/.venv/<name>` with `--system-site-packages`.

**EnsembleLauncher:**
```bash
cd envs/el
bash create_env.sh
source $HOME/.venv/el/bin/activate
```

**Parsl** (used in weak and strong scaling experiments):
```bash
cd envs/parsl
bash create_env.sh
source $HOME/.venv/parsl/bin/activate
```

**Dask** (used in weak and strong scaling experiments):
```bash
cd envs/dask
bash create_env.sh
source $HOME/.venv/dask/bin/activate
```

**Parsl + Dask** (used in microbenchmark experiments):
```bash
cd envs/parsl_dask
bash create_env.sh
source $HOME/.venv/parsl_dask/bin/activate
```

## Experiments

### Weak Scaling (`experiments/weakscaling/`)

Measures performance as the number of nodes increases while keeping work per node constant. Each framework variant runs the same workload across node counts from 2 to 8192.

**Framework variants:** `el`, `el_cluster`, `el_cluster_1level_ws`, `el_cluster_2level_ws`, `mpi`, `parsl_with_trigger`, `dask_1level_with_trigger`, `dask_2level_processpool_with_trigger`

**How to run:**
1. Initialize directories for all node counts:
   ```bash
   cd experiments/weakscaling/el
   bash setup_dir.sh        # creates per-node directories (2, 4, 8, ..., 8192)
   ```
2. Submit jobs for each node count:
   ```bash
   cd <node_count>
   qsub submit_<fmwork>.sh
   ```

The submit scripts loop over trials, sleep times, and task counts. A checkpoint system (`checkpoints/done_*`) prevents re-running completed configurations on resubmission. To rerun a specific configuration, delete its checkpoint file (e.g., `rm checkpoints/done_102_10_100ms_2levels_1`).

**Sweep parameters:**

| Parameter | Values | Description |
|-----------|--------|-------------|
| Sleep time (ms) | 0, 1, 10, 100, 1000, 60000 | Task work duration (busy-wait spin loop) |
| Tasks per worker | 10 | Number of tasks submitted per worker |
| Concurrent workers | 102 | CPU cores used per node (Aurora has 104; cores 0 and 52 are reserved) |
| Trials | 3 | Repeated runs per configuration |
| Node counts | 2 to 8192 | Varies by framework (EL up to 8192, Parsl up to 2048, Dask up to 256) |

Total tasks per run = tasks_per_worker x concurrent_workers x num_nodes.

**Directory-to-label mapping:**

| Directory | Plot Label |
|-----------|------------|
| `el` | EL |
| `el_cluster` | EL(Cluster) |
| `el_cluster_1level_ws` | EL(Pull, Depth=1) |
| `el_cluster_2level_ws` | EL(Pull) |
| `mpi` | MPI |
| `parsl_with_trigger` | Parsl |
| `dask_1level_with_trigger` | Dask(Depth=0) |
| `dask_2level_processpool_with_trigger` | Dask(Depth=1) |

**EL variant differences:**

- **`el`** — Direct launcher with `nlevels=2`, no cluster mode. Tasks are submitted as a dictionary to `EnsembleLauncher.run()`.
- **`el_cluster`** — Cluster mode enabled. Uses a `ClusterClient` to submit tasks dynamically via `client.submit_batch()`. Adds `result_buffer_size=10000` and `result_flush_interval=0.05`.
- **`el_cluster_1level_ws`** — 1-level hierarchy (`nlevels=1`) with work-stealing enabled. Configures `task_request_size=102` and `task_request_interval=0.5s`.
- **`el_cluster_2level_ws`** — 2-level hierarchy (`nlevels=2`) with work-stealing enabled. Uses a slower `task_request_interval=1.0s` and explicit heartbeat settings (`heartbeat_interval=10s`, `heartbeat_dead_threshold=120s`).

**Other framework notes:**

- **`mpi`** — Pure MPI baseline. Launches `mpiexec -n <total_ranks> --ppn 102` with a task wrapper script.
- **`parsl_with_trigger`** — Uses a Future-based trigger (`--trigger`) to decouple task submission from execution start, allowing measurement of submission overhead.
- **`dask_*_with_trigger`** — Uses a trigger file (`--trigger`) to synchronize task execution start. Disables Dask work-stealing and memory management via environment variables. The `1level` variant uses a flat worker pool; `2level_processpool` adds a process pool layer.

**Queue mapping** (set automatically by `setup_dir.sh`):
- ≤ 2 nodes: `debug`
- ≤ 16 nodes: `capacity`
- ≤ 128 nodes: `debug-scaling`
- \> 128 nodes: `prod`

### Strong Scaling (`experiments/strongscaling/`)

Measures performance with a fixed total amount of work as parallelism increases. The total task count is fixed at 208,896 (= 1 x 102 x 2048) regardless of how many nodes are used.

**Framework variants:** `el`, `el_cluster`, `el_cluster_2level_ws`, `mpi`, `parsl_with_trigger`, `dask_2level_processpool_with_trigger`

The run procedure is the same as weak scaling: `setup_dir.sh` to initialize, `qsub submit.sh` per node count, then `merge_timelines.py` to aggregate results.

**Differences from weak scaling:**

| Parameter | Weak Scaling | Strong Scaling |
|-----------|-------------|----------------|
| Sleep times (ms) | 0, 1, 10, 100, 1000, 60000 | 1000, 60000 |
| Tasks per worker | 10 | 1 |
| Total tasks | Scales with node count | Fixed at 208,896 |
| Node counts | 2–8192 | 64–2048 |

The EL variant configurations are the same as in weak scaling. The `el_cluster_1level_ws` and `dask_1level_with_trigger` variants are not included in strong scaling. The directory-to-label mapping is the same as in weak scaling.

### Microbenchmarks (`experiments/microbenchmarks/`)

Low-level measurements of framework overhead. Uses the combined `parsl_dask` environment.

**Roundtrip Latency** (`roundtrip_latency/`):
Measures per-task submission and completion time across frameworks (ProcessPool, EL, Parsl, Dask).
```bash
qsub submit.sh
```
Produces per-task timing CSVs in `data/`.

**Worker Throughput** (`worker_throughput/`):
Measures task throughput (tasks/sec) under sequential and concurrent submission.
```bash
qsub submit.sh
```
Produces throughput CSVs in `data/`.

### Flexibility (`experiments/flexibility/`)

Evaluates custom scheduling policies and multi-stage pipeline workloads.

**Parametric Sweep** (`parametric_sweep/`):
Compares four scheduling policies (FIFO, shortest-first, longest-first, largest-first) under task distributions with varying coefficient of variation.
```bash
qsub submit.sh
```
Custom policies are defined in `policies/custom_policies.py`.

**Pipeline** (`pipeline/`):
Runs multi-stage pipeline workloads (e.g., inference + simulation stages) with per-stage resource requirements defined in `configs/` JSON files.
```bash
qsub submit.sh
```

## Experiment Workflow Summary

```
setup_dir.sh                   # Initialize per-node directories
    -> qsub submit.sh          # Submit PBS job (loops over trials, sleep times, task counts)
        -> test.py             # Run benchmark, write logs to logs/
        -> merge_timelines.py  # Aggregate timeline CSVs from all nodes via MPI
        -> Results archived to all_logs/logs_<config>/
    -> post/plot_fig_X.py      # Read all_logs, generate PDF figures
```

## Regenerating Plots

All plotting scripts are in `experiments/*/post/` directories. They read experiment log data and produce publication-quality PDF figures using `matplotlib` and `scienceplots`.

**Dependencies:** `matplotlib`, `scienceplots`, `numpy`, `pandas`

**To regenerate a figure**, `cd` into the corresponding `post/` directory and run the script:
```bash
cd experiments/weakscaling/post
python plot_fig_6a.py
```

### Figure-to-Script Mapping

| Figure  | Script                                              | Output              |
|---------|-----------------------------------------------------|---------------------|
| Fig 1   | `experiments/weakscaling/post/plot_fig_1.py`        | `figs/fig_1.pdf`    |
| Fig 4   | `experiments/microbenchmarks/post/plot_fig_4.py`    | `fig_4.pdf`         |
| Fig 5   | `experiments/microbenchmarks/post/plot_fig_5.py`    | `fig_5.pdf`         |
| Fig 6a  | `experiments/weakscaling/post/plot_fig_6a.py`       | `figs/fig_6a.pdf`   |
| Fig 6b  | `experiments/strongscaling/post/plot_fig_6b.py`     | `fig_6b.pdf`        |
| Fig 7   | `experiments/weakscaling/post/plot_fig_7.py`        | `figs/fig_7.pdf`    |
| Fig 8   | `experiments/weakscaling/post/plot_fig_8.py`        | `figs/fig_8.pdf`    |
| Fig 9   | `experiments/weakscaling/post/plot_fig_9.py`        | `figs/fig_9.pdf`    |
| Fig 10  | `experiments/flexibility/post/plot_fig_10.py`       | `fig_10.pdf`        |
| Fig 11  | `experiments/flexibility/post/plot_fig_11.py`       | `fig_11.pdf`        |
| Table 1 | `experiments/weakscaling/post/plot_table_1.py`      | `data/table_1.csv`  |

**Notes:**
- Fig 9 accepts `--node-count`, `--nworkers`, `--ntasks-per-worker`, and `--sleep-time` arguments.
- Fig 11 accepts `--policy` and trial-related arguments.
- Some scripts expect `figs/` or `data/` subdirectories to exist in the `post/` directory. If you get a "No such file or directory" error, create them with `mkdir -p figs data`.
