#!/bin/bash -l
#PBS -l select=32
#PBS -l walltime=01:00:00
#PBS -q debug-scaling
#PBS -A datascience
#PBS -l filesystems=home:flare
#PBS -N mofa


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

echo "*** Running EnsembleLauncher pipeline benchmark ***"

NNODES=$(wc -l < "$PBS_NODEFILE")

NPIPELINES=120000

 # "fixed_leafs_children_policy" "resource_split_policy"; do
for i in $(seq 1 1); do
    for POLICY in "routing_policy";do
        ##VARS
        ext="${POLICY}_npipe${NPIPELINES}_${i}"
        LOG_DIR=logs_mofa_$ext
        CKPT="checkpoints/done_mofa_${ext}"
        CONFIG="configs/mofa_config.json"
        ##
        if [ -f ${CKPT} ]; then
            continue
        fi
        rm -r ${LOG_DIR}
        # Clean up any leftover timeline files on all nodes
        mpirun -np "$NNODES" --ppn 1 bash -c 'rm -f /tmp/timeline_*.csv /tmp/mpi_timeline_*.csv'
        ##CMD
        python3 benchmark_async.py --config ${CONFIG} --npipelines $NPIPELINES --policy $POLICY --log_dir ${LOG_DIR} --use_tags
        ##merge timelines
        mpirun -np "$NNODES" --ppn 1 python3 merge_timelines.py --policy $POLICY --iter $i --log_dir ${LOG_DIR}
        touch $CKPT
        if [ -d "all_logs/${LOG_DIR}" ]; then
            rm -r all_logs/${LOG_DIR}
        fi
        mv ${LOG_DIR} all_logs/
    done
done

echo "Benchmark complete"
