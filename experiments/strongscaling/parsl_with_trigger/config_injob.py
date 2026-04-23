import os
from parsl.config import Config
from parsl.addresses import address_by_interface

# This is just one example config, please see the Aurora documentation for parsl for more
# Config versions and options: https://docs.alcf.anl.gov/aurora/workflows/parsl/

# Use LocalProvider to launch workers within a submitted batch job
from parsl.providers import LocalProvider
# The high throughput executor is for scaling large single core/tile/gpu tasks on HPC system:
from parsl.executors import HighThroughputExecutor
# Use the MPI launcher to launch worker processes:
from parsl.launchers import MpiExecLauncher


def get_num_nodes():
    """Get the number of nodes from PBS_NODEFILE."""
    try:
        node_file = os.getenv("PBS_NODEFILE")
        with open(node_file, "r") as f:
            node_list = f.readlines()
            return len(node_list)
    except (FileNotFoundError, TypeError) as e:
        print(f"Warning: Could not read PBS_NODEFILE: {e}")
        return 1


def get_high_throughput_executor(
    label,
    num_nodes,
    max_workers_per_node,
    available_accelerators=None,
    cpu_affinity=None,
    prefetch_capacity=0
):
    """
    Create a HighThroughputExecutor configuration.
    
    Args:
        label: Executor label (e.g., "cpu", "gpu")
        num_nodes: Number of nodes in the job
        max_workers_per_node: Maximum workers per node
        available_accelerators: List of accelerator names (optional)
        cpu_affinity: CPU affinity string (optional)
        prefetch_capacity: Prefetch capacity for tasks (default: 0)
    
    Returns:
        HighThroughputExecutor instance
    """
    executor_kwargs = {
        "label": label,
        "address": address_by_interface('bond0'),
        "max_workers_per_node": max_workers_per_node,
        "prefetch_capacity": prefetch_capacity,
        "provider": LocalProvider(
            nodes_per_block=num_nodes,
            launcher=MpiExecLauncher(bind_cmd="--cpu-bind", overrides="--ppn 1"),
            init_blocks=1,
            max_blocks=1,
        ),
    }
    
    if available_accelerators is not None:
        executor_kwargs["available_accelerators"] = available_accelerators
    
    if cpu_affinity is not None:
        executor_kwargs["cpu_affinity"] = cpu_affinity
    
    return HighThroughputExecutor(**executor_kwargs)


def create_config(executors):
    """
    Create a Parsl Config with the given executors.
    
    Args:
        executors: List of HighThroughputExecutor instances
    
    Returns:
        Config instance
    """
    return Config(
        executors=executors,
        initialize_logging=False
    )


# Setup affinity and accelerator configurations
all_cores = [f"{i}" for i in range(104)]
all_cores.pop(52)
all_cores.pop(0)
start_cores = [1, 9, 17, 25, 33, 41, 53, 61, 69, 77, 85, 93]


# Get the number of nodes
num_nodes = get_num_nodes()

# Build configurations
aurora_single_core_tile_config = create_config([
    get_high_throughput_executor(
        label="cpu",
        num_nodes=num_nodes,
        max_workers_per_node=102,
        cpu_affinity="list:" + ":".join(all_cores),
        prefetch_capacity=0
    ),
    get_high_throughput_executor(
        label="gpu",
        num_nodes=num_nodes,
        max_workers_per_node=12,
        available_accelerators=[f"{i}" for i in range(12)],
        cpu_affinity="list:" + ":".join([f"{st}-{st+7}" for st in start_cores]),
        prefetch_capacity=0
    ),
])

aurora_single_core_config = create_config([
    get_high_throughput_executor(
        label="cpu",
        num_nodes=num_nodes,
        max_workers_per_node=102,
        cpu_affinity="list:" + ":".join(all_cores),
        prefetch_capacity=0
    ),
])



aurora_single_tile_config = create_config([
    get_high_throughput_executor(
        label="gpu",
        num_nodes=num_nodes,
        max_workers_per_node=12,
        available_accelerators=[f"{i}" for i in range(12)],
        cpu_affinity="list:" + ":".join([f"{st}-{st+7}" for st in start_cores]),
        prefetch_capacity=0
    ),
])

aurora_two_ccs_config = create_config([
    get_high_throughput_executor(
        label="gpu",
        num_nodes=num_nodes,
        max_workers_per_node=24,
        available_accelerators=[f"{i}" for i in range(12) for _ in range(2)],
        cpu_affinity="list:" + ":".join([f"{st}-{st+3}:{st+4}-{st+7}" for st in start_cores]),
        prefetch_capacity=0
    ),
])

aurora_four_ccs_config = create_config([
    get_high_throughput_executor(
        label="gpu",
        num_nodes=num_nodes,
        max_workers_per_node=48,
        available_accelerators=[f"{i}" for i in range(12) for _ in range(4)],
        cpu_affinity="list:" + ":".join([f"{st}-{st+1}:{st+2}-{st+3}:{st+4}-{st+5}:{st+6}-{st+7}" for st in start_cores]),
        prefetch_capacity=0
    ),
])


def get_nworker_config(nworkers:int=64):
    any_nworker_config = create_config([
        get_high_throughput_executor(
            label="cpu",
            num_nodes=num_nodes,
            max_workers_per_node=nworkers,
            prefetch_capacity=0
        ),
    ])

    return any_nworker_config



