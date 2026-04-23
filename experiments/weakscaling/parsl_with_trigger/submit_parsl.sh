#!/bin/bash -l
#PBS -l select=<node>
#PBS -l walltime=01:00:00
#PBS -q <queue>
#PBS -A datascience
#PBS -l filesystems=home:flare



if [ -n "${PBS_O_WORKDIR}" ]; then
    cd ${PBS_O_WORKDIR}
fi
export ZE_FLAT_DEVICE_HIERARCHY=FLAT

source $HOME/.envs/parsl/bin/activate

dirname=$(pwd)
export PYTHONPATH=${dirname}:${PYTHONPATH}
echo working_dir: $dirname
mkdir -p all_logs
mkdir -p timings
mkdir -p checkpoints

echo "*** Running Parsl performance tests ***"
# Get number of nodes
NUM_NODES=$(wc -l < ${PBS_NODEFILE})
echo "Number of nodes: ${NUM_NODES}"

for i in $(seq 1 3); do
    for SLEEPTIME_MS in 0 1 10 100 1000 60000; do
        for NUMTASKS in 10; do
            for NWORKERS in 102; do
                # Calculate total tasks: tasks_per_worker * concurrent_workers * num_nodes
                TOTAL_TASKS=$((NUMTASKS * NWORKERS * NUM_NODES))
                
                # Convert milliseconds to seconds for Python script
                SLEEPTIME_SEC=$(echo "scale=6; ${SLEEPTIME_MS} / 1000" | bc)
                
                ext=${NWORKERS}_${NUMTASKS}_${SLEEPTIME_MS}ms_${i}
                [ -f "checkpoints/done_${ext}" ] && continue

                # Clean up old timeline files from /tmp on all nodes before each test
                echo "Cleaning old timeline files from /tmp..."
                mpirun -np ${NUM_NODES} --ppn 1 bash -c 'rm -f /tmp/timeline_*.csv /tmp/debug_task_*.txt'

                START_TIME=$(date +%s.%N)
                echo "* Run Parsl test: workers=${NWORKERS}, tasks_per_worker=${NUMTASKS}, total_tasks=${TOTAL_TASKS}, sleep=${SLEEPTIME_MS}ms (${SLEEPTIME_SEC}s), iter=${i} *"
                LAUNCH_SCRIPT=${dirname}/test.py
                EXE="python ${LAUNCH_SCRIPT} --sleep-time=${SLEEPTIME_SEC} --num-tasks=${TOTAL_TASKS} --concurrent-workers=${NWORKERS} --track-timeline --trigger"
                echo ${EXE}
                ${EXE}
                END_TIME=$(date +%s.%N)
                ELAPSED=$(echo "$END_TIME - $START_TIME" | bc)
                echo "Elapsed time for test: ${ELAPSED} seconds" >> timings/elapsed_time_${NWORKERS}_${NUMTASKS}_${SLEEPTIME_MS}ms.txt
                echo ""

                # Check if all tasks finished successfully before creating checkpoint
                if grep -q "All .* tasks finished in .* seconds" logs/main.log; then
                    touch "checkpoints/done_${ext}"
                    mpirun -np ${NUM_NODES} python3 merge_timelines.py
                    [ -d "all_logs/logs_${ext}" ] && rm -rf all_logs/logs_${ext}
                    mv logs all_logs/logs_${ext}
                    mv runinfo all_logs/logs_${ext}/
                else
                    echo "Warning: Task completion message not found in logs/main.log for ${ext}"
                    [ -d "all_logs/logs_${ext}" ] && rm -rf all_logs/logs_${ext}
                    mv logs all_logs/logs_${ext}
                    mv runinfo all_logs/logs_${ext}/
                fi
            done
        done
    done
done
