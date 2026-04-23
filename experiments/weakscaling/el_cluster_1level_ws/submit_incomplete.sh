#!/bin/bash
# Submit jobs for node counts that don't have all done files yet.
# Each submit script handles its own restart logic internally.
# Usage: submit_incomplete.sh <username>

if [ $# -lt 1 ]; then
    echo "Error: username argument required." >&2
    echo "Usage: $0 <username>" >&2
    exit 1
fi
USERNAME=$1

BASE=/lus/flare/projects/datascience/EL-SC26/el_cluster_1level_ws

BATCH_SIZES=(0 510 1020)
REPEATS=(1 2 3)
MAX_NODES=128

is_in_queue() {
    local dir=${1%/}   # strip trailing slash
    [ "${IN_QUEUE_DIRS[$dir]+_}" ]
}

is_complete() {
    local dir=$1
    for batch in "${BATCH_SIZES[@]}"; do
        for rep in "${REPEATS[@]}"; do
            local f="${dir}/checkpoints/done_102_10_1000ms_1levels_${batch}_${rep}"
            if [ ! -f "$f" ]; then
                return 1
            fi
        done
    done
    return 0
}

while true; do

    # Build a set of directories that already have jobs in the queue/running.
    # We extract PBS_O_WORKDIR from qstat -f for each job belonging to ht1410.
    unset IN_QUEUE_DIRS
    declare -A IN_QUEUE_DIRS
    while IFS= read -r jobid; do
        workdir=$(qstat -f "$jobid" 2>/dev/null \
            | tr -d '\t\n' \
            | grep -oP 'PBS_O_WORKDIR=\K[^,]+')
        if [ -n "$workdir" ]; then
            IN_QUEUE_DIRS["$workdir"]=1
        fi
    done < <(qstat -u "$USERNAME" 2>/dev/null | awk 'NR>5 && $1 ~ /^[0-9]/ {split($1,a,"."); print a[1]}')

    echo "=== el_cluster_1level_ws (max nodes: ${MAX_NODES}) ==="

    for node_dir in $(ls -d ${BASE}/*/  2>/dev/null | sort -t/ -k9 -n); do
        n=$(basename $node_dir)
        # skip non-numeric dirs
        [[ "$n" =~ ^[0-9]+$ ]] || continue
        # skip above max
        [ "$n" -gt "$MAX_NODES" ] && continue

        if is_complete "$node_dir"; then
            echo "  $n nodes: complete"
        elif is_in_queue "${node_dir%/}"; then
            echo "  $n nodes: incomplete — already in queue, skipping"
        else
            echo "  $n nodes: incomplete — submitting"
            (cd "$node_dir" && qsub submit_el.sh)
        fi
    done

    echo Iteration done. Sleeping......

    sleep 3600
done
