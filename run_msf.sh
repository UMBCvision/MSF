#!/usr/bin/env bash

set -x
set -e

CUDA_VISIBLE_DEVICES=$1 python train_msf.py \
    --cos \
    --learning_rate 0.05 \
    --epochs 200 \
    --arch resnet50 \
    --topk 5 \
    --momentum 0.99 \
    --mem_bank_size 1024000 \
    --augmentation 'weak/weak' \
    --checkpoint_path output/msf_1_ww_topk_5_mbs_1024k_resnet50 \
    /datasets/imagenet

