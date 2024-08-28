#!/bin/bash

start=$1
end=$2
job_index=$3

echo "Job index: $job_index, start: $start, end: $end"

# Inner loop from start to end
for (( j=start; j<=end; j++ ))
do
    python dataset-scripts/tracking-scripts/save_pick_tasks.py --data-shard $j --data-dir /ariesdv0/zhanling/oxe-data-converted --obj-data-dir /ariesdv0/zhanling/oxe-data-converted/fractal20220817_pick_data/0.1.0
done