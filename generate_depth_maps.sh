#!/bin/bash

# Outer loop from 0 to 7
for job_index in {0..7}
do
    # Calculate start and end values for the inner loop
    start=$(( (1024 / 8) * job_index ))
    end=$(( (1024 / 8) * (job_index + 1) -1 ))

    echo "Job index: $job_index, start: $start, end: $end"

    # Inner loop from start to end
    for (( j=start; j<=end; j++ ))
    do
        python get_depth_maps.py --data-shard $j --use-metric-depth-model --batch-size 4 --data-dir /ariesdv0/zhanling/oxe-data-converted --checkpoint-path /ariesdv0/zhanling/checkpoints
        python save_depth_maps.py --data-shard $j --data-dir /ariesdv0/zhanling/oxe-data-converted --depth-data-dir /ariesdv0/zhanling/oxe-data-converted/fractal20220817_depth_data/0.1.0
    done
done
