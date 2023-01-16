#!/bin/bash


TMPDIR = "/data/test_cityscapes/cityscapes"

# copy assets
# rsync -aP /save_data/hrnetv2_w48_imagenet_pretrained.pth ${TMPDIR}/hrnetv2_w48_imagenet_pretrained.pth

# define scratch dir
SCRATCH_DIR="/save_data"

# TMPDIR: data dir, SCRATCH_DIR: save_dir
sh run_h_48_d_4_prob_proto.sh train 'hrnet_proto_80k' ${TMPDIR} ${SCRATCH_DIR}
