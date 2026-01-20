#!/bin/bash

data_path=$1

if [ -z "$data_path" ]; then
    echo "Error: No data_path provided."
    echo "Usage: $0 <data_path>"
    exit 1
fi

for num_points in 5000 10000 15000 20000 25000 30000 35000 40000 45000 50000 55000 60000 65000 70000 75000 80000 85000 90000 95000 100000 393216
do
CUDA_VISIBLE_DEVICES=0 python train.py -d $data_path \
--data_name kodak_small --model_name GaussianImage_Cholesky --num_points $num_points --iterations 50000 --save_imgs
done
