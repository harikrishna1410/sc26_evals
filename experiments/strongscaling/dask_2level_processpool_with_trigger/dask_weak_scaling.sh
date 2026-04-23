#!/bin/bash -l
#PBS -l select=<node>
#PBS -l walltime=01:00:00
#PBS -q <queue>
#PBS -A datascience
#PBS -l filesystems=home:flare
#PBS -N strong_dask

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


for i in $(seq 1 3); do
    for SLEEPTIME_MS in 1000 60000; do
        for NUMTASKS in 1; do
            for CONCURRENT_WORKERS in 102; do  # Add more: 24 48 102
                # Calculate total tasks: tasks_per_worker * concurrent_workers * num_nodes
                TOTAL_TASKS=$((NUMTASKS * CONCURRENT_WORKERS * 2048))
                
                # Convert milliseconds to seconds for Python script
                SLEEPTIME_SEC=$(echo "scale=6; ${SLEEPTIME_MS} / 1000" | bc)
                
                ext=${CONCURRENT_WORKERS}_${NUMTASKS}_${SLEEPTIME_MS}ms_${i}
                [ -f "checkpoints/done_${ext}" ] && continue

                [ -f "dask_trigger_file" ] && rm dask_trigger_file

                # Clean up old timeline files from /tmp on all nodes before each test
                echo "Cleaning old timeline files from /tmp..."
                mpirun -np ${NUM_NODES} --ppn 1 bash -c 'rm -f /tmp/timeline_*.csv /tmp/debug_task_*.txt'
                
                START_TIME=$(date +%s.%N)
                echo "* Run Dask test: workers=${CONCURRENT_WORKERS}, tasks_per_worker=${NUMTASKS}, total_tasks=${TOTAL_TASKS}, sleep=${SLEEPTIME_MS}ms (${SLEEPTIME_SEC}s), iter=${i} *"
                EXE="python ${dirname}/test_dask.py --sleep-time=${SLEEPTIME_SEC} --num-tasks=${TOTAL_TASKS} --concurrent-workers=${CONCURRENT_WORKERS} --scheduler-mode=manual --track-timeline --one-worker-per-node --processpool --trigger"
                echo ${EXE}
                ${EXE} 
                END_TIME=$(date +%s.%N)
                ELAPSED=$(echo "$END_TIME - $START_TIME" | bc)
                echo "Elapsed time for test: ${ELAPSED} seconds" >> timings/elapsed_time_${CONCURRENT_WORKERS}_${NUMTASKS}_${SLEEPTIME_MS}ms.txt
                echo ""
                
                pkill -f dask ##kill any background dask tasks

                if [ -f "logs/main.log" ] && grep -q "All tasks finished in .* seconds" logs/main.log; then
                    touch checkpoints/done_${ext}
                    [ -d "all_logs/logs_${ext}" ] && rm -rf all_logs/logs_${ext}
                    mpirun -np ${NUM_NODES} python3 merge_timelines.py
                    mv logs all_logs/logs_${ext}
                else
                    echo "Warning: Test did not complete successfully (no completion message in logs/main.log)"
                    [ -d "all_logs/logs_${ext}" ] && rm -rf all_logs/logs_${ext}
                    mv logs all_logs/logs_${ext}
                fi
                
            done
        done
    done
done

echo "Benchmark complete"
