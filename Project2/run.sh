#!/bin/bash

# List of models
models=("CNN" "UNet" "UNet_base")

# List of optimizers
optimizers=("adam" "sgd")

# List of loss functions
loss_functions=("bce_loss" "dice_loss" "focal_loss" "bce_total_variation")

# Set your CUDA device here
export CUDA_VISIBLE_DEVICES=0

# Loop over the models
for model in ${models[@]}; do

  # Loop over the optimizers
  for optimizer in ${optimizers[@]}; do

    # Loop over the loss functions
    for loss_function in ${loss_functions[@]}; do

      # Command to run the model with different configurations
      python main.py \
        --n_features 16 \
        --model_type ${model} \
        --data_augmentation \
        --batch_size 2 \
        --num_workers 8 \
        --optimizer_type ${optimizer} \
        --lr_scheduler reducelronplateau \
        --early_stopping \
        --early_stopping_patience 18 \
        --lr 1e-3 \
        --num_epochs 100 \
        --loss_function ${loss_function} \
        --data_choice Retina

    done
  done
done