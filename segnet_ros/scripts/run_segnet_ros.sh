#!/usr/bin/env bash
cd /home/catkin_ws/src/segnet_ros/src

DATA_ROOT="/data"
SCRATCH_ROOT="/data"
SINGLE_SCALE="ss"

DATA_DIR="${DATA_ROOT}/HKUSTGZ"
BACKBONE="hrnet48"

CONFIGS="/home/catkin_ws/src/segnet/configs/hkustgz/hkustgz_ros.json"
CONFIGS_TEST="/home/catkin_ws/src/segnet/configs/hkustgz/hkustgz_ros.json"

MODEL_NAME="hr_w48_attn_uncer_proto"
LOSS_TYPE="pixel_uncer_prototype_ce_loss"
CHECKPOINTS_ROOT="/data/checkpoints"
CHECKPOINTS_NAME="hr_w48_attn_uncer_proto_hkustgz_max_performance.pth"
LOG_FILE="${SCRATCH_ROOT}/logs/hkustgz/${CHECKPOINTS_NAME}.log"
echo "Logging to $LOG_FILE"
mkdir -p `dirname $LOG_FILE`

PRETRAINED_MODEL="/save_data/hrnetv2_w48_imagenet_pretrained.pth"
MAX_ITERS=40000
BATCH_SIZE=20
BASE_LR=0.003

echo "[single scale] test"
  python3 -u inference.py --configs ${CONFIGS} \
                        --backbone ${BACKBONE} --model_name ${MODEL_NAME} --checkpoints_name ${CHECKPOINTS_NAME} \
                        --phase test_ros --gpu 0 \
                        --test_dir ${DATA_DIR}/test --log_to_file n \


