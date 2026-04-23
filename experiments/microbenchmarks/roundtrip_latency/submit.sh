#!/bin/bash -l
#PBS -l select=1
#PBS -l walltime=05:00:00
#PBS -q capacity
#PBS -A datascience
#PBS -l filesystems=home:flare



if [ -n "${PBS_O_WORKDIR}" ]; then
    cd ${PBS_O_WORKDIR}
fi
export ZE_FLAT_DEVICE_HIERARCHY=FLAT

source $HOME/.envs/parsl_dask/bin/activate

for i in 4; do
    NTASKS=$((10**i))
    # for sleeptime in 0.0 0.001 0.01 0.1 1.0; do
    for sleeptime in 0.0; do
        python3 latency_experiment.py --ntasks=$NTASKS --output-file="data/latency_results_${NTASKS}_sequential.csv" --sleeptime $sleeptime --framework "all" --sequential
    done
done