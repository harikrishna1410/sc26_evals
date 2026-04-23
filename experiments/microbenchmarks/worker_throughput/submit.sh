#!/bin/bash -l
#PBS -l select=1
#PBS -l walltime=01:00:00
#PBS -q debug
#PBS -A datascience
#PBS -l filesystems=home:flare



if [ -n "${PBS_O_WORKDIR}" ]; then
    cd ${PBS_O_WORKDIR}
fi
export ZE_FLAT_DEVICE_HIERARCHY=FLAT

source $HOME/.venv/parsl_dask/bin/activate

for i in 1 2 3 4 5; do
    NTASKS=$((10**i))
    for sleeptime in 0.0 0.001 0.01 0.1 1.0; do
        python3 throughput_experiment.py --ntasks=$NTASKS --output-file="data/throughput_results_v2.csv" --sleeptime $sleeptime --framework "all"
    done
done