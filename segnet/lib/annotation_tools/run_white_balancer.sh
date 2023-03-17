#!/usr/bin/env bash

export PYTHONWARNINGS="ignore"

OUTPUT_DIR = "/data/Data/kin/ruoyu_data/HKUSTGZ/frame00"
INPUT_DIR = "/data/Data/jjiao/dataset/FusionPortable_dataset_develop/sensor_data/mini_hercules/20230309_hkustgz_campus_road_day/data/frame_cam00/image/data"

python3 -u -m debugpy --listen 5678 --wait-for-client image_white_balancer.py 
        --output_dir ${OUTPUT_DIR}
        --input_dir ${INPUT_DIR} 