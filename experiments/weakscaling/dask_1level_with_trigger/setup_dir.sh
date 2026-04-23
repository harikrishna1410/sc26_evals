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
files_to_link=(test_dask.py config_dask.py merge_timelines.py utils.py)

for node in ${nodes[@]};do
    if [ -d $node ]; then
        # Always remove checkpoints
        rm -rf $node/checkpoints
        rm -rf $node/logs
        
        if [ "$RESET_TYPE" == "hard" ]; then
            # Hard reset: delete timings and all_logs
            rm -rf $node/timings
            rm -rf $node/all_logs
        else
            # Soft reset: move all_logs and timings to all_logs_{i} and timings_{i}
            if [ -d $node/all_logs ] || [ -d $node/timings ]; then
                i=0
                while [ -d "$node/all_logs_$i" ] || [ -d "$node/timings_$i" ]; do
                    i=$((i + 1))
                done
                
                if [ -d $node/all_logs ]; then
                    echo "Moving $node/all_logs to $node/all_logs_$i"
                    mv $node/all_logs $node/all_logs_$i
                fi
                
                if [ -d $node/timings ]; then
                    echo "Moving $node/timings to $node/timings_$i"
                    mv $node/timings $node/timings_$i
                fi
            fi
        fi
    else
        mkdir $node
    fi

    for f in ${files_to_copy[@]};do
        cp $f $node/$f
    done

    for f in ${files_to_copy[@]};do
        sed -i "s/<node>/$node/g" $node/$f
        if [ $node -le 16 ]; then
            sed -i "s/<queue>/capacity/g" $node/$f
        elif [ $node -le 128 ]; then
            sed -i "s/<queue>/debug-scaling/g" $node/$f
        else
            sed -i "s/<queue>/prod/g" $node/$f
        fi
    done

    for f in ${files_to_link[@]};do
        rm $node/$f
        ln -s ../$f $node/$f
    done
done