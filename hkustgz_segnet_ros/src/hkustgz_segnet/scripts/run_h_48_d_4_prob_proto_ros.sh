#!/usr/bin/env bash
cd /home/hkustgz_segnet/hkustgz_segnet_ros/src/hkustgz_segnet/src

DATA_ROOT="/data"
SCRATCH_ROOT="/save_data"
ASSET_ROOT=${DATA_ROOT}
SINGLE_SCALE="ss"

DATA_DIR="${DATA_ROOT}/Cityscapes"
SAVE_DIR="${SCRATCH_ROOT}/Cityscapes/seg_results/"
BACKBONE="hrnet48"

CONFIGS="/home/hkustgz_segnet/hkustgz_segnet/configs/cityscapes/H_48_D_4_prob_proto.json"
CONFIGS_TEST="/home/hkustgz_segnet/hkustgz_segnet/configs/cityscapes/H_48_D_4_TEST.json"

MODEL_NAME="hr_w48_prob_proto"
LOSS_TYPE="pixel_prob_prototype_ce_loss"
CHECKPOINTS_ROOT="${SCRATCH_ROOT}/Cityscapes"
CHECKPOINTS_NAME="${MODEL_NAME}_lr1x_fast_mls_loss"
LOG_FILE="${SCRATCH_ROOT}/logs/Cityscapes/${CHECKPOINTS_NAME}.log"
echo "Logging to $LOG_FILE"
mkdir -p `dirname $LOG_FILE`

PRETRAINED_MODEL="/save_data/hrnetv2_w48_imagenet_pretrained.pth"
MAX_ITERS=40000
BATCH_SIZE=20
BASE_LR=0.003

echo "[single scale] test"
  python3 -u -m debugpy --listen 5678 --wait-for-client inference.py --configs ${CONFIGS} \
                        --backbone ${BACKBONE} --model_name ${MODEL_NAME} --checkpoints_name ${CHECKPOINTS_NAME} \
                        --phase test_ros --gpu 1 --resume ${CHECKPOINTS_ROOT}/checkpoints/cityscapes/${CHECKPOINTS_NAME}_max_performance.pth \
                        --test_dir ${DATA_DIR}/test --log_to_file n \
                        --out_dir ${SAVE_DIR}${CHECKPOINTS_NAME}_test_ss



