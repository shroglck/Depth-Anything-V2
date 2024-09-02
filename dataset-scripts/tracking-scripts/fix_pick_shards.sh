#!/bin/bash
# Inner loop from start to end
for (( j=0; j<=1023; j++ ))
do
    python dataset-scripts/tracking-scripts/fix_pick_shards.py --data-shard $j
done