#!/bin/bash

start=$1
end=$2
job_index=$3

echo "Job index: $job_index, start: $start, end: $end"

# Inner loop from start to end
for (( j=start; j<=end; j++ ))
do
    python dataset-scripts/get_key_objects.py --data-shard $j --data-dir /ariesdv0/openvla/OXE/rlds_mod
    python dataset-scripts/save_key_objects.py --data-shard $j --data-dir /ariesdv0/openvla/OXE/rlds_mod --obj-data-dir /ariesdv0/openvla/OXE/rlds_mod/fractal20220817_obj_data/0.1.0
done
