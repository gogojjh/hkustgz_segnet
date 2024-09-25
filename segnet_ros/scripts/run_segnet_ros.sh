#!/usr/bin/env bash
PATH_SEGNET="/Titan/code/mapping_ws/src/cobra/src/HKUSTGZ_SegNet/segnet"
PATH_SEGNET_ROS="/Titan/code/mapping_ws/src/cobra/src/HKUSTGZ_SegNet/segnet_ros"
cd "${PATH_SEGNET_ROS}/src"

DATA_ROOT="/Spy/dataset/cobra_results/cobra_seg_results"
SCRATCH_ROOT="/Spy/dataset/cobra_results/cobra_seg_results"
SINGLE_SCALE="ss"

DATA_DIR="${DATA_ROOT}/HKUSTGZ"
BACKBONE="hrnet48"

# TODO by users
CONFIGS="${PATH_SEGNET}/configs/hkust/hkust_ros.json"
CONFIGS_TEST="${PATH_SEGNET}/configs/hkust/hkust_ros.json"

MODEL_NAME="hr_w48_attn_uncer_proto"
LOSS_TYPE="pixel_uncer_prototype_ce_loss"
CHECKPOINTS_ROOT="${DATA_ROOT}/checkpoints/hkustgz"
CHECKPOINTS_NAME="cs_fs.pth"
LOG_FILE="${SCRATCH_ROOT}/logs/hkust/${CHECKPOINTS_NAME}.log"
echo "Logging to $LOG_FILE"
mkdir -p `dirname $LOG_FILE`

PRETRAINED_MODEL="/save_data/hrnetv2_w48_imagenet_pretrained.pth"
MAX_ITERS=40000
BATCH_SIZE=20
BASE_LR=0.003

# NOTE(gogojjh): add by jjiao, do not need to change hkustgz_ros.json
export PYTHONPATH="${PYTHONPATH}:${PATH_SEGNET}"

echo "[single scale] test"
  python3 -u inference.py --configs ${CONFIGS} \
                          --backbone ${BACKBONE} \
                          --model_name ${MODEL_NAME} \
                          --checkpoints_name ${CHECKPOINTS_NAME} \
                          --phase test_ros --gpu 0 \
                          --test_dir ${DATA_DIR}/test --log_to_file n \
