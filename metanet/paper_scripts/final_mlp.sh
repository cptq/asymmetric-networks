#!/bin/bash

# Prompt the user for the dataset name
dataset_name=${1:-"resnet8"}
max_size=${2:-10000}

echo "Dataset: $dataset_name"
echo "Model: MLP"
echo "Dataset size: $max_size"

if [ "$dataset_name" == "resnet8" ] || [ "$dataset_name" == "smaller_resnet8" ]; then
    lr=1e-2
elif [ "$dataset_name" == "sparse_resnet8" ]; then
    lr=1e-4
fi


for i in {1..5}
do
    echo "Running training iteration $i"
    echo "Learning rate: $lr"
    python train.py --model mlp --dataset "$dataset_name" --max_size $max_size --lr $lr
done

echo "Training completed."
