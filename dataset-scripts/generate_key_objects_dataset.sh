#!/bin/bash

start=$1
end=$2
job_index=$3
api_key=$4

echo "Job index: $job_index, start: $start, end: $end"

# Inner loop from start to end
for (( j=start; j<=end; j++ ))
do
    python dataset-scripts/get_key_objects.py --data-shard $j --data-dir /ariesdv0/zhanling/oxe-data-converted --openai-key ${api_key}
    python dataset-scripts/save_key_objects.py --data-shard $j --data-dir /ariesdv0/zhanling/oxe-data-converted --obj-data-dir /ariesdv0/zhanling/oxe-data-converted/fractal20220817_obj_data/0.1.0
done

# Save positional words into single file
python dataset-scripts gather_positional_words.py --data-dir /ariesdv0/zhanling/oxe-data-converted