#!/bin/bash

# Parse command line arguments
RESET_TYPE=${1:-soft}

if [[ "$RESET_TYPE" != "hard" && "$RESET_TYPE" != "soft" ]]; then
    echo "Usage: $0 [hard|soft]"
    echo "  hard: Delete checkpoints and timings (default)"
    echo "  soft: Delete checkpoints and move timings to timings_{i}"
    exit 1
fi

echo "Running $RESET_TYPE reset"

nodes=(2 8 32 64 128 256 512 1024 2048 4096 8192)

files_to_copy=(submit_mpiexec.sh)
files_to_link=(utils.py task_wrapper.sh)

for node in ${nodes[@]};do
    if [ -d $node ]; then
        # Always remove checkpoints and output files
        rm -rf $node/checkpoints
        rm -f $node/*.e* $node/*.o*
        
        if [ "$RESET_TYPE" == "hard" ]; then
            # Hard reset: delete timings
            rm -rf $node/timings
        else
            # Soft reset: move timings to timings_{i}
            if [ -d $node/timings ]; then
                i=0
                while [ -d "$node/timings_$i" ]; do
                    i=$((i + 1))
                done
                
                echo "Moving $node/timings to $node/timings_$i"
                mv $node/timings $node/timings_$i
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
        rm -f $node/$f
        ln -s ../$f $node/$f
    done
done