RUN="bash generate_depth_maps.sh"
JOB="DepthAnythingV2"
aries run -j 8 -g 1 -n aries-b04 ag-${JOB} lingzhan/openvla -- bash -c "nvidia-smi && git clone https://github.com/akshaygopalkr/Depth-Anything-V2.git && cd Depth-Anything-V2 && ${RUN}"