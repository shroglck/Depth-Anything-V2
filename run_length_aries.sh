RUN="bash get_dataset_length.py --data-dir /ariesdv0/zhanling/oxe-data-converted"

aries run -g 0 lingzhan/openvla -- bash -c "nvidia-smi && git clone https://github.com/akshaygopalkr/Depth-Anything-V2.git && cd Depth-Anything-V2 && ${RUN}"