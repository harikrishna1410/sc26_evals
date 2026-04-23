#!/bin/bash -l
#PBS -l select=<node>
#PBS -l walltime=01:00:00
#PBS -q <queue>
#PBS -A datascience
#PBS -l filesystems=home:flare



if [ -n "${PBS_O_WORKDIR}" ]; then
    cd ${PBS_O_WORKDIR}
fi

module load frameworks

export ZE_FLAT_DEVICE_HIERARCHY=FLAT

# Get number of nodes
NUM_NODES=`wc -l < $PBS_NODEFILE`
echo "Number of nodes: ${NUM_NODES}"

dirname=$(pwd)
echo working_dir: $dirname
mkdir -p timings
mkdir -p checkpoints

echo "*** Running MPI performance tests ***"
for i in $(seq 1 3); do
    for SLEEPTIME_MS in 0 1 10 100 1000 60000; do
        for NUMTASKS in 10; do
            for CONCURRENT_WORKERS in 102; do
                # Calculate total tasks and ranks: tasks_per_worker * concurrent_workers * num_nodes
                TOTAL_TASKS=$((NUMTASKS * CONCURRENT_WORKERS * NUM_NODES))
                TOTAL_RANKS=$((CONCURRENT_WORKERS * NUM_NODES))
                TASKS_PER_RANK=$NUMTASKS
                
                # Convert milliseconds to seconds for the application
                SLEEPTIME_SEC=$(echo "scale=6; ${SLEEPTIME_MS} / 1000" | bc)
                
                ext=${CONCURRENT_WORKERS}_${NUMTASKS}_${SLEEPTIME_MS}ms_${i}
                [ -f "checkpoints/done_${ext}" ] && continue
                START_TIME=$(date +%s.%N)
                echo "* Run MPI test: workers=${CONCURRENT_WORKERS}, tasks_per_worker=${NUMTASKS}, total_tasks=${TOTAL_TASKS}, total_ranks=${TOTAL_RANKS}, sleep=${SLEEPTIME_MS}ms (${SLEEPTIME_SEC}s), iter=${i} *"
                mpiexec -n ${TOTAL_RANKS} --ppn ${CONCURRENT_WORKERS} ./task_wrapper.sh ${TASKS_PER_RANK} ${SLEEPTIME_SEC}
                END_TIME=$(date +%s.%N)
                ELAPSED=$(echo "$END_TIME - $START_TIME" | bc)
                echo "Elapsed time for test: ${ELAPSED} seconds" >> timings/elapsed_time_${CONCURRENT_WORKERS}_${NUMTASKS}_${SLEEPTIME_MS}ms.txt
                echo ""
                
                # Create checkpoint after successful run
                touch "checkpoints/done_${ext}"
                mpirun -np ${NUM_NODES} python3 merge_timelines.py
            done
        done
    done
done

echo "Benchmark complete"
