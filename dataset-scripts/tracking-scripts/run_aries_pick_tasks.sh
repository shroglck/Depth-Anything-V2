JOB="rt1-pick-key-objects"
RUN="bash dataset-scripts/tracking-scripts/save_pick_tasks.sh" 

for job_index in {0..15}
do  
    start=$(( (1024 / 16) * job_index ))
    end=$(( (1024 / 16) * (job_index + 1) -1 ))
    aries run ag-${JOB}-${job_index} -j 1 -g 0 lingzhan/openvla -- bash -c "git clone https://github.com/akshaygopalkr/Depth-Anything-V2.git && cd Depth-Anything-V2 && ${RUN} ${start} ${end} ${job_index}"
done
