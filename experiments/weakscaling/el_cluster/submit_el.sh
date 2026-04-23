#!/bin/bash -l
#PBS -l select=<node>
#PBS -l walltime=01:00:00
#PBS -q <queue>
#PBS -A datascience
#PBS -l filesystems=home:flare
#PBS -N cluster_el



if [ -n "${PBS_O_WORKDIR}" ]; then
    cd ${PBS_O_WORKDIR}
fi
export ZE_FLAT_DEVICE_HIERARCHY=FLAT

source $HOME/.envs/el/bin/activate

dirname=$(pwd)
echo working_dir: $dirname
mkdir -p checkpoints
mkdir -p timings
mkdir -p all_logs

echo "*** Running EnsembleLauncher performance tests ***"
# Get number of nodes
NUM_NODES=$(wc -l < ${PBS_NODEFILE})
echo "Number of nodes: ${NUM_NODES}"

for i in $(seq 1 3); do
    for SLEEPTIME_MS in 0 1 10 100 1000 60000; do
        for NUMTASKS in 10; do
            for nlevels in 2; do
                for CONCURRENT_WORKERS in 102; do
                    # Calculate total tasks: tasks_per_worker * concurrent_workers * num_nodes
                    TOTAL_TASKS=$((NUMTASKS * CONCURRENT_WORKERS * NUM_NODES))
                    
                    # Convert milliseconds to seconds for Python script
                    SLEEPTIME_SEC=$(echo "scale=6; ${SLEEPTIME_MS} / 1000" | bc)

                    ext=${CONCURRENT_WORKERS}_${NUMTASKS}_${SLEEPTIME_MS}ms_${nlevels}levels_${i}
                    [ -f "checkpoints/done_${ext}" ] && continue
                    START_TIME=$(date +%s.%N)
                    echo "* Run EL test: workers=${CONCURRENT_WORKERS}, tasks_per_worker=${NUMTASKS}, total_tasks=${TOTAL_TASKS}, sleep=${SLEEPTIME_MS}ms (${SLEEPTIME_SEC}s), nlevels=${nlevels}, iter=${i} *"
                    EXE="python ${dirname}/test.py --sleep-time=${SLEEPTIME_SEC} --num-tasks=${TOTAL_TASKS} --concurrent-workers=${CONCURRENT_WORKERS} --nlevels=${nlevels}"
                    echo ${EXE}
                    ${EXE}
                    END_TIME=$(date +%s.%N)
                    ELAPSED=$(echo "$END_TIME - $START_TIME" | bc)
                    echo "Elapsed time for test: ${ELAPSED} seconds" >> timings/elapsed_time_${CONCURRENT_WORKERS}_${NUMTASKS}_${SLEEPTIME_MS}ms_${nlevels}levels.txt
                    echo ""

                    # Check for successful completion before marking as done
                    if grep -q "All tasks finished in .* seconds" logs/script.log; then
                        touch checkpoints/done_${ext}

                        mpirun -np ${NUM_NODES} python3 merge_timelines.py
                        # Move status and results into logs directory
                        [ -f "main_status.json" ] && mv main_status.json logs/
                        [ -f "results.json" ] && mv results.json logs/
                        [ -d "profiles" ] && mv profiles logs/

                        # Move logs directory to all_logs
                        [ -d "all_logs/logs_${ext}" ] && rm -rf all_logs/logs_${ext}
                        mv logs all_logs/logs_${ext}
                    else
                        echo "Warning: Test did not complete successfully (no completion message in logs/main.log)"
                        [ -f "main_status.json" ] && mv main_status.json logs/
                        [ -f "results.json" ] && mv results.json logs/
                        [ -d "profiles" ] && mv profiles logs/

                        # Move logs directory to all_logs
                        [ -d "all_logs/logs_${ext}" ] && rm -rf all_logs/logs_${ext}
                        mv logs all_logs/logs_${ext}
                    fi
                done
            done
        done
    done
done

echo "Benchmark complete"