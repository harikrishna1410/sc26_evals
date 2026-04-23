#!/bin/bash

# Parse command line arguments
RESET_TYPE=${1:-soft}

if [[ "$RESET_TYPE" != "hard" && "$RESET_TYPE" != "soft" ]]; then
    echo "Usage: $0 [hard|soft]"
    echo "  hard: Delete checkpoints and timings (default)"
    echo "  soft: Delete checkpoints and move all_logs to all_logs_{i}"
    exit 1
fi

echo "Running $RESET_TYPE reset"

nodes=(2 4 8 16 32 64 128 256)

files_to_copy=(dask_weak_scaling.sh)

for node in ${nodes[@]};do
    if [ -d $node ]; then
        for f in ${files_to_copy[@]};do
            cp $f $node/$f
        done

        for f in ${files_to_copy[@]};do
            sed -i "s/<node>/$node/g" $node/$f
            if [ $node -le 16 ]; then
                sed -i "s/<queue>/capacity/g" $node/$f
            elif [ $node -le 255 ]; then
                sed -i "s/<queue>/debug-scaling/g" $node/$f
            else
                sed -i "s/<queue>/prod/g" $node/$f
            fi
        done
    fi
done