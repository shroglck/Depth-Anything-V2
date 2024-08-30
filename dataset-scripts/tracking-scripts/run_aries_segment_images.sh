JOB="rt1-segment-key-objects"
INSTALL="git clone https://github.com/facebookresearch/segment-anything-2.git && cd segment-anything-2 && pip install -e ."
INSTALL2="cd checkpoints && ./download_ckpts.sh && cd ../.."
RUN="bash dataset-scripts/tracking-scripts/track_key_objects_pick.sh" 

for job_index in {0..7}
do  
    start=$(( (1024 / 8) * job_index ))
    end=$(( (1024 / 8) * (job_index + 1) -1 ))
    aries run ag-${JOB}-${job_index} -j 1 -g 1 lingzhan/openvla -- bash -c "nvidia-smi && git clone https://github.com/akshaygopalkr/Depth-Anything-V2.git && cd Depth-Anything-V2 && ${INSTALL} && ${INSTALL2} && ${RUN} ${start} ${end} ${job_index}"
done