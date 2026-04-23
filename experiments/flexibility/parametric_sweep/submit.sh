#!/bin/bash -l
#PBS -l select=16
#PBS -l walltime=01:00:00
#PBS -q debug-scaling
#PBS -A datascience
#PBS -l filesystems=home:flare
#PBS -N parametric_sweep



if [ -n "${PBS_O_WORKDIR}" ]; then
    cd ${PBS_O_WORKDIR}
fi
export ZE_FLAT_DEVICE_HIERARCHY=FLAT

source $HOME/.venv/el/bin/activate

dirname=$(pwd)
echo working_dir: $dirname
mkdir -p checkpoints
mkdir -p all_logs
rm -r logs

echo "*** Running EnsembleLauncher flexibility tests ***"

NNODES=$(wc -l < "$PBS_NODEFILE")

POLICIES="fifo_policy,shortest_first,longest_first,largest_first"
for i in $(seq 1 15); do
    for VAR in 9 81 225 576 900 3600;do
        ext="${POLICIES}_${VAR}_${i}"
        if [ -f "checkpoints/done_${ext}" ]; then
            continue
        fi
        # Clean up any leftover timeline files on all nodes
        mpirun -np "$NNODES" --ppn 1 bash -c 'rm -f /tmp/mpi_timeline_*.csv'
        python3 benchmark.py --ntasks 100 --nnodes "1,4" --variance_sec $VAR --policy ${POLICIES}
        touch "checkpoints/done_${ext}"
        if [ -d "all_logs/logs_${ext}" ]; then
            rm -r all_logs/logs_${ext}
        fi
        mv logs all_logs/logs_${ext}
    done
done

echo "Benchmark complete"