#!/bin/bash

# Prompt the user for the dataset name
dataset_name=${1:-"resnet8"}
max_size=${2:-10000}

echo "Dataset: $dataset_name"
echo "Model: DeepSets"
echo "Dataset size: $max_size"

lr_lst=(1e-5 1e-4 5e-4 1e-3 5e-3 1e-2)

for lr in "${lr_lst[@]}"
do
    echo "Learning rate: $lr"
    python train.py --model deepsets --hidden_dim 64 --dataset "$dataset_name" --max_size $max_size --lr $lr
done

echo "Training completed."
