API_KEY=$1
JOB="bridge-key-objects"
RUN="python dataset-scripts/get_key_objects_bridge.py --data-dir /ariesdv0/zhanling/oxe-data-converted --openai-api-key ${API_KEY}"
aries run ag-${JOB} -j 1 -g 0 lingzhan/openvla -- bash -c "pip install openai && git clone https://github.com/akshaygopalkr/Depth-Anything-V2.git && cd Depth-Anything-V2 && ${RUN}"