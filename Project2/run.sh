#!/bin/bash

# List of models
models=("UNet")

# List of optimizers
optimizers=("adam")

# List of loss functions
loss_functions=("bce_loss" "dice_loss" "bce_total_variation")

# Set your CUDA device here
export CUDA_VISIBLE_DEVICES=0

# Set the flag to decide whether to use data augmentation or not
use_data_augmentation=true

# Check if data augmentation flag is set
if [ "$use_data_augmentation" = true ]; then
  data_augmentation_flag="--data_augmentation"
else
  data_augmentation_flag=""
fi

# Loop over the models
for model in ${models[@]}; do

  # Loop over the optimizers
  for optimizer in ${optimizers[@]}; do

    # Loop over the loss functions
    for loss_function in ${loss_functions[@]}; do

      # Command to run the model with different configurations
      python main.py \
        --n_features 24 \
        --model_type ${model} \
        ${data_augmentation_flag} \
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