#!/bin/bash -l
#PBS -l select=2
#PBS -l walltime=00:10:00
#PBS -q prod
#PBS -A datascience
#PBS -l filesystems=home:flare

if [ -n "${PBS_O_WORKDIR}" ]; then
    cd ${PBS_O_WORKDIR}
fi

# Load necessary modules
module load frameworks

# Activate conda environment if needed
source ~/.envs/dask/bin/activate

# Set environment variables
export DASK_DISTRIBUTED__SCHEDULER__WORK_STEALING=False
export DASK_DISTRIBUTED__WORKER__MEMORY__TARGET=False
export DASK_DISTRIBUTED__WORKER__MEMORY__SPILL=False
export DASK_DISTRIBUTED__WORKER__MEMORY__PAUSE=False
export DASK_DISTRIBUTED__WORKER__MEMORY__TERMINATE=False

dirname=$(pwd)
export PYTHONPATH=${dirname}:${PYTHONPATH}
echo working_dir: $dirname
mkdir -p timings
mkdir -p all_logs
mkdir -p checkpoints

echo "*** Running Dask performance tests ***"
# Get number of nodes
NUM_NODES=$(wc -l < ${PBS_NODEFILE})
echo "Number of nodes: ${NUM_NODES}"

CONCURRENT_WORKERS=64
NUMTASKS=1
TOTAL_TASKS=$((NUMTASKS * CONCURRENT_WORKERS * NUM_NODES))
SLEEPTIME_SEC=1.0

python ${dirname}/test_dask.py --sleep-time=${SLEEPTIME_SEC} --num-tasks=${TOTAL_TASKS} --concurrent-workers=${CONCURRENT_WORKERS} --scheduler-mode=manual --track-timeline --one-worker-per-node

mpirun -np ${NUM_NODES} python3 merge_timelines.py