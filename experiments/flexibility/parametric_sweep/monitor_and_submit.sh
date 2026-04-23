#!/bin/bash

JOB_NAME="parametri"
USER="ht1410"
SLEEP_INTERVAL=1800

while true; do
    # Count jobs with the given name in the queue
    job_count=$(qstat -u "$USER" 2>/dev/null | grep "$JOB_NAME" | wc -l)

    if [ "$job_count" -lt 2 ]; then
        echo "$(date): Only $job_count '$JOB_NAME' job(s) in queue (< 2). Submitting new job..."
        qsub submit_el.sh
        echo "$(date): Job submitted."
    else
        echo "$(date): $job_count '$JOB_NAME' job(s) in queue. Skipping submission."
    fi

    echo "$(date): Sleeping for $SLEEP_INTERVAL seconds..."
    sleep "$SLEEP_INTERVAL"
done
