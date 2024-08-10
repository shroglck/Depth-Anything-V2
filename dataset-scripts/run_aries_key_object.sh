JOB="rt1-key-objects"
OPENAPI_KEY=a210b026e24949e7a1711022f4c1856e
RUN="echo export OPENAPI_KEY=\"${OPENAPI_KEY}\" >> ~/.bashrc && source ~/.bashrc && bash generate_key_objects_dataset.sh" 

for job_index in {0..7}
do  

    start=$(( (1024 / 8) * job_index ))
    end=$(( (1024 / 8) * (job_index + 1) -1 ))
    aries run ag-${JOB}-${job_index} -j 1 -g 0 lingzhan/openvla -- bash -c "nvidia-smi && git clone https://github.com/akshaygopalkr/Depth-Anything-V2.git && cd Depth-Anything-V2 && ${RUN} ${start} ${end} ${job_index}"
done
