RUN="bash generate_depth_maps.sh"
JOB="DepthAnythingV2"
aries run -j 8 -g 1 -n aries-b03,aries-b04,aries-b05,aries-c01,aries-c02,aries-c03,aries-c04,aries-c05 ag-${JOB} lingzhan/openvla -- bash -c "git clone https://github.com/akshaygopalk
r/Depth-Anything-V2.git && cd Depth-Anything-V2 && ${RUN}"