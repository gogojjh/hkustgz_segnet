#!/usr/bin/env bash
SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

export PYTHONWARNINGS="ignore"

cd $SCRIPTPATH
cd ../../

DATA_ROOT=$3
SCRATCH_ROOT=$4
ASSET_ROOT=${DATA_ROOT}

DATA_DIR="/data/HKUSTGZ"
SAVE_DIR="${SCRATCH_ROOT}/hkustgz/seg_results/"
BACKBONE="hrnet48"

CONFIGS="configs/hkustgz/v3/fusionportable_cityscapes_v3.json"
CONFIGS_TEST="configs/hkustgz/H_48_D_4_TEST.json"

MODEL_NAME="hr_w48_attn_uncer_proto"
LOSS_TYPE="pixel_uncer_prototype_ce_loss"
CHECKPOINTS_ROOT="${SCRATCH_ROOT}/hkustgz"
CHECKPOINTS_NAME="${MODEL_NAME}_hkustgz"
LOG_FILE="${SCRATCH_ROOT}/logs/hkustgz/${CHECKPOINTS_NAME}.log"
echo "Logging to $LOG_FILE"
mkdir -p `dirname $LOG_FILE`

PRETRAINED_MODEL="/save_data/hrnetv2_w48_imagenet_pretrained.pth"
MAX_ITERS=160000
BATCH_SIZE=16
BASE_LR=0.01

if [ "$1"x == "train"x ]; then
  python3 -u main.py --configs ${CONFIGS} \
                       --drop_last y \
                       --phase train \
                       --gathered n \
                       --loss_balance y \
                       --log_to_file n \
                       --backbone ${BACKBONE} \
                       --model_name ${MODEL_NAME} \
                       --gpu 0 1\
                       --data_dir ${DATA_DIR} \
                       --loss_type ${LOSS_TYPE} \
                       --max_iters ${MAX_ITERS} \
                       --checkpoints_root ${CHECKPOINTS_ROOT} \
                       --checkpoints_name ${CHECKPOINTS_NAME} \
                       --pretrained ${PRETRAINED_MODEL} \
                       --train_batch_size ${BATCH_SIZE} \
                       --distributed \
                       --base_lr ${BASE_LR} \
                       2>&1 | tee ${LOG_FILE}


elif [ "$1"x == "resume"x ]; then
  python3 -u -m debugpy --listen 5678 --wait-for-client main.py --configs ${CONFIGS} \
                       --drop_last y \
                       --phase train \
                       --gathered n \
                       --loss_balance y \
                       --log_to_file n \
                       --backbone ${BACKBONE} \
                       --model_name ${MODEL_NAME} \
                       --max_iters ${MAX_ITERS} \
                       --data_dir ${DATA_DIR} \
                       --loss_type ${LOSS_TYPE} \
                       --gpu 0 1\
                       --checkpoints_root ${CHECKPOINTS_ROOT} \
                       --checkpoints_name ${CHECKPOINTS_NAME} \
                       --resume_continue y \
                       --train_batch_size ${BATCH_SIZE} \
                       --distributed \
                        2>&1 | tee -a ${LOG_FILE}


elif [ "$1"x == "val"x ]; then
  python3 -u main.py --configs ${CONFIGS} --drop_last y  --data_dir ${DATA_DIR} \
                       --backbone ${BACKBONE} --model_name ${MODEL_NAME} --checkpoints_name ${CHECKPOINTS_NAME} \
                       --phase test --gpu 0 1 2 3 \
                       --loss_type ${LOSS_TYPE} --test_dir ${DATA_DIR}/val/image \
                       --out_dir ${SAVE_DIR}${CHECKPOINTS_NAME}_val_ms

  python3 -m lib.metrics.cityscapes_evaluator --pred_dir ${SAVE_DIR}${CHECKPOINTS_NAME}_val_ms/label  \
                                       --gt_dir ${DATA_DIR}/val/label

elif [ "$1"x == "test"x ]; then
  if [ "$5"x == "ss"x ]; then
    echo "[single scale] test"
    python3 -u main.py --configs ${CONFIGS} --drop_last y --data_dir ${DATA_DIR} \
                         --backbone ${BACKBONE} --model_name ${MODEL_NAME} --checkpoints_name ${CHECKPOINTS_NAME} \
                         --phase test --gpu 0 1 \
                         --test_dir ${DATA_DIR}/test --log_to_file n \
                         --out_dir ${SAVE_DIR}${CHECKPOINTS_NAME}_test_ss
  else
    echo "[multiple scale + flip] test"
    python3 -u main.py --configs ${CONFIGS_TEST} --drop_last y --data_dir ${DATA_DIR} \
                         --backbone ${BACKBONE} --model_name ${MODEL_NAME} --checkpoints_name ${CHECKPOINTS_NAME} \
                         --phase test --gpu 0 1 2 3 \
                         --test_dir ${DATA_DIR}/test --log_to_file n \
                         --out_dir ${SAVE_DIR}${CHECKPOINTS_NAME}_test_ms
  fi

elif [ "$1"x == "test_ros"x ]; then
  if [ "$5"x == "ss"x ]; then
    echo "[single scale] test"
    python3 -u /hkustsegnet_ros/inference.py --configs ${CONFIGS} --drop_last y --data_dir ${DATA_DIR} \
                         --backbone ${BACKBONE} --model_name ${MODEL_NAME} --checkpoints_name ${CHECKPOINTS_NAME} \
                         --phase test \
                         --gpu 0 1 2 3 \
                         --test_dir ${DATA_DIR}/test --log_to_file n \
                         --out_dir ${SAVE_DIR}${CHECKPOINTS_NAME}_test_ss
  else
    echo "[multiple scale + flip] test"
    python3 -u /hkustsegnet_ros/inference.py --configs ${CONFIGS_TEST} --drop_last y --data_dir ${DATA_DIR} \
                         --backbone ${BACKBONE} --model_name ${MODEL_NAME} --checkpoints_name ${CHECKPOINTS_NAME} \
                         --phase test --gpu 0 1 2 3 \
                         --test_dir ${DATA_DIR}/test --log_to_file n \
                         --out_dir ${SAVE_DIR}${CHECKPOINTS_NAME}_test_ms
  fi

else
  echo "$1"x" is invalid..."
fi
