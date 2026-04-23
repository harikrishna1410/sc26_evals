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

nodes=(64 128 256 512 1024 2048)

files_to_copy=(submit_el.sh)
files_to_link=(utils.py test.py merge_timelines.py)

for node in ${nodes[@]};do
    if [ -d $node ]; then
        # Always remove checkpoints, logs and PBS output files
        rm -rf $node/checkpoints
        # rm -rf $node/.ckpt*
        rm -rf $node/logs
        rm -rf $node/profiles
        rm -f $node/*.e*
        rm -f $node/*.o*
        
        if [ "$RESET_TYPE" == "hard" ]; then
            # Hard reset: delete timings, all_logs, and all_profiles
            rm -rf $node/timings
            rm -rf $node/all_logs
            rm -rf $node/all_profiles
        else
            # Soft reset: move all_logs, timings, and all_profiles to versioned directories
            if [ -d $node/all_logs ] || [ -d $node/timings ] || [ -d $node/all_profiles ]; then
                i=0
                while [ -d "$node/all_logs_$i" ] || [ -d "$node/timings_$i" ] || [ -d "$node/all_profiles_$i" ]; do
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
                
                if [ -d $node/all_profiles ]; then
                    echo "Moving $node/all_profiles to $node/all_profiles_$i"
                    mv $node/all_profiles $node/all_profiles_$i
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
        if [ $node -le 2 ]; then
            sed -i "s/<queue>/debug/g" $node/$f
        elif [ $node -le 16 ]; then
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