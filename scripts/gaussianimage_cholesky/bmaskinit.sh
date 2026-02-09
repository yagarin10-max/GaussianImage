#!/bin/bash

BASE_PATH="/mnt/t/DATA/Bedroom_20"

IMAGE_PATH="${BASE_PATH}/images"
MASK_PATH="${BASE_PATH}/binary_mask"

for num_points in 12000
do
CUDA_VISIBLE_DEVICES=0 python train.py -d $IMAGE_PATH \
--data_name binary \
--model_name GaussianImage_Cholesky \
--num_points $num_points \
--iterations 50000 \
--save_imgs \
--use_wandb \
--mask_dataset $MASK_PATH \
--match_mask_points \
--no_clamp
done

for num_points in 78100
do
CUDA_VISIBLE_DEVICES=0 python train.py -d $IMAGE_PATH \
--data_name binary \
--model_name GaussianImage_Cholesky \
--num_points $num_points \
--iterations 50000 \
--save_imgs \
--use_wandb \
--no_clamp
done