#!/bin/bash

data_path=$1

if [ -z "$data_path" ]; then
    echo "Error: No data_path provided."
    echo "Usage: $0 <data_path>"
    exit 1
fi

for num_points in 50000 60000 # 768x512
do
CUDA_VISIBLE_DEVICES=0 python train.py -d $data_path \
--data_name kodak \
--model_name GaussianImage_Cholesky_wMask \
--num_points $num_points \
--iterations 50000 \
--save_imgs \
--start_mask_training 5000 \
--stop_mask_training 20000 \
--reg_type kl \
--target_sparsity 0.2 \
--lambda_reg 0.05 \
--init_mask_logit 1.0
done

for num_points in 50000 60000 # 768x512
do
CUDA_VISIBLE_DEVICES=0 python train.py -d $data_path \
--data_name kodak \
--model_name GaussianImage_Cholesky_wMask \
--num_points $num_points \
--iterations 50000 \
--save_imgs \
--start_mask_training 5000 \
--stop_mask_training 20000 \
--reg_type kl \
--target_sparsity 0.2 \
--lambda_reg 0.05 \
--init_mask_logit 2.0
done

for num_points in 50000 60000 # 768x512
do
CUDA_VISIBLE_DEVICES=0 python train.py -d $data_path \
--data_name kodak \
--model_name GaussianImage_Cholesky_wMask \
--num_points $num_points \
--iterations 50000 \
--save_imgs \
--start_mask_training 5000 \
--stop_mask_training 20000 \
--reg_type kl \
--target_sparsity 0.2 \
--lambda_reg 0.05 \
--init_mask_logit 3.0
done

for num_points in 50000 60000 # 768x512
do
CUDA_VISIBLE_DEVICES=0 python train.py -d $data_path \
--data_name kodak \
--model_name GaussianImage_Cholesky_wMask \
--num_points $num_points \
--iterations 50000 \
--save_imgs \
--start_mask_training 5000 \
--stop_mask_training 20000 \
--reg_type kl \
--target_sparsity 0.3 \
--lambda_reg 0.05 \
--init_mask_logit 1.0
done

for num_points in 50000 60000 # 768x512
do
CUDA_VISIBLE_DEVICES=0 python train.py -d $data_path \
--data_name kodak \
--model_name GaussianImage_Cholesky_wMask \
--num_points $num_points \
--iterations 50000 \
--save_imgs \
--start_mask_training 5000 \
--stop_mask_training 20000 \
--reg_type kl \
--target_sparsity 0.3 \
--lambda_reg 0.05 \
--init_mask_logit 2.0
done

for num_points in 50000 60000 # 768x512
do
CUDA_VISIBLE_DEVICES=0 python train.py -d $data_path \
--data_name kodak \
--model_name GaussianImage_Cholesky_wMask \
--num_points $num_points \
--iterations 50000 \
--save_imgs \
--start_mask_training 5000 \
--stop_mask_training 20000 \
--reg_type kl \
--target_sparsity 0.3 \
--lambda_reg 0.05 \
--init_mask_logit 3.0
done

for num_points in 50000 60000 # 768x512
do
CUDA_VISIBLE_DEVICES=0 python train.py -d $data_path \
--data_name kodak \
--model_name GaussianImage_Cholesky_wMask \
--num_points $num_points \
--iterations 50000 \
--save_imgs \
--start_mask_training 5000 \
--stop_mask_training 20000 \
--reg_type kl \
--target_sparsity 0.2 \
--lambda_reg 0.05 \
--init_mask_logit 1.0 \
--no_clamp
done

for num_points in 50000 60000 # 768x512
do
CUDA_VISIBLE_DEVICES=0 python train.py -d $data_path \
--data_name kodak \
--model_name GaussianImage_Cholesky_wMask \
--num_points $num_points \
--iterations 50000 \
--save_imgs \
--start_mask_training 5000 \
--stop_mask_training 20000 \
--reg_type kl \
--target_sparsity 0.2 \
--lambda_reg 0.05 \
--init_mask_logit 2.0 \
--no_clamp
done

for num_points in 50000 60000 # 768x512
do
CUDA_VISIBLE_DEVICES=0 python train.py -d $data_path \
--data_name kodak \
--model_name GaussianImage_Cholesky_wMask \
--num_points $num_points \
--iterations 50000 \
--save_imgs \
--start_mask_training 5000 \
--stop_mask_training 20000 \
--reg_type kl \
--target_sparsity 0.2 \
--lambda_reg 0.05 \
--init_mask_logit 3.0 \
--no_clamp
done

for num_points in 50000 60000 # 768x512
do
CUDA_VISIBLE_DEVICES=0 python train.py -d $data_path \
--data_name kodak \
--model_name GaussianImage_Cholesky_wMask \
--num_points $num_points \
--iterations 50000 \
--save_imgs \
--start_mask_training 5000 \
--stop_mask_training 20000 \
--reg_type kl \
--target_sparsity 0.3 \
--lambda_reg 0.05 \
--init_mask_logit 1.0 \
--no_clamp
done

for num_points in 50000 60000 # 768x512
do
CUDA_VISIBLE_DEVICES=0 python train.py -d $data_path \
--data_name kodak \
--model_name GaussianImage_Cholesky_wMask \
--num_points $num_points \
--iterations 50000 \
--save_imgs \
--start_mask_training 5000 \
--stop_mask_training 20000 \
--reg_type kl \
--target_sparsity 0.3 \
--lambda_reg 0.05 \
--init_mask_logit 2.0 \
--no_clamp
done

for num_points in 50000 60000 # 768x512
do
CUDA_VISIBLE_DEVICES=0 python train.py -d $data_path \
--data_name kodak \
--model_name GaussianImage_Cholesky_wMask \
--num_points $num_points \
--iterations 50000 \
--save_imgs \
--start_mask_training 5000 \
--stop_mask_training 20000 \
--reg_type kl \
--target_sparsity 0.3 \
--lambda_reg 0.05 \
--init_mask_logit 3.0 \
--no_clamp
done