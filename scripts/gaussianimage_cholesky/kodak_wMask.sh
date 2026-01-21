#!/bin/bash

data_path=$1

if [ -z "$data_path" ]; then
    echo "Error: No data_path provided."
    echo "Usage: $0 <data_path>"
    exit 1
fi

for num_points in 70000 # 768x512
do
CUDA_VISIBLE_DEVICES=0 python train.py -d $data_path \
--data_name kodak \
--model_name GaussianImage_Cholesky_wMask \
--num_points $num_points \
--iterations 50000 \
--save_imgs \
--start_mask_training 5000 \
--stop_mask_training 10000 \
--use_wandb \
--reg_type kl \
--target_sparsity 0.7 \
--lambda_reg 0.005 \
--init_mask_logit 2.0
done

for num_points in 70000 # 768x512
do
CUDA_VISIBLE_DEVICES=0 python train.py -d $data_path \
--data_name kodak \
--model_name GaussianImage_Cholesky_wMask \
--num_points $num_points \
--iterations 50000 \
--save_imgs \
--start_mask_training 5000 \
--stop_mask_training 10000 \
--use_wandb \
--reg_type kl \
--target_sparsity 0.7 \
--lambda_reg 0.005 \
--init_mask_logit 1.0
done
