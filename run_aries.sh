RUN="bash generate_depth_maps.sh"
JOB="DepthAnything"

for job_index in {0..7}
do  

    start=$(( (1024 / 8) * job_index ))
    end=$(( (1024 / 8) * (job_index + 1) -1 ))
    aries run -j 1 -g 1 ag-${JOB}-${job_index} lingzhan/openvla -- bash -c "git clone https://github.com/akshaygopalkr/Depth-Anything-V2.git && cd Depth-Anything-V2 && ${RUN} ${start} ${end} ${job_index}"
done