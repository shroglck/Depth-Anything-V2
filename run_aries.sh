RUN="bash generate_depth_maps.sh"
JOB="DepthAnythingV2"
aries run -j 4 -g 1 -n aries-b04 ag-${JOB} lingzhan/openvla -- bash -c "nvidia-smi && ls -la . && ${RUN}"