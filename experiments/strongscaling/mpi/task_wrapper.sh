#!/bin/bash

set -eu -o pipefail

RANK=${PMIX_RANK:-0}
NTASKS_PER_RANK=${1:-}
SLEEPTIME_SEC=${2:-}

if [ -z "$NTASKS_PER_RANK" ] || [ -z "$SLEEPTIME_SEC" ]; then
  echo "Usage: $0 NTASKS_PER_RANK SLEEPTIME_SEC"
  exit 1
fi

# Run utils.py to execute the tasks for this rank
python3 utils.py --sleep_time=${SLEEPTIME_SEC} --task_id=${RANK} --ntasks=${NTASKS_PER_RANK}
