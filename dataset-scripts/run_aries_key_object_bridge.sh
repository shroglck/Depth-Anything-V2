API_KEY=$1
JOB="bridge-key-objects"
RUN="python dataset-scripts/track_key_objects_bridge.py --data-dir /ariesdv0/zhanling/oxe-data-converted --api-key ${API_KEY}"
aries run ag-${JOB}-${job_index} -j 1 -g 0 lingzhan/openvla -- bash -c "nvidia-smi && git clone https://github.com/akshaygopalkr/Depth-Anything-V2.git && cd Depth-Anything-V2 && ${RUN} ${start} ${end} ${job_index}"