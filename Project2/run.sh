#!/bin/bash

echo "Running models on Retina dataset"
CUDA_VISIBLE_DEVICES=0 python main.py --model_type=CNN --data_augmentation --optimizer_type=adam --lr_scheduler=reducelronplateau --loss_function=bce_loss --batch_size=8 --data_choice=Retina --num_epochs=20
CUDA_VISIBLE_DEVICES=0 python main.py --model_type=CNN --data_augmentation --optimizer_type=adam --lr_scheduler=reducelronplateau --loss_function=dice_loss --batch_size=8 --data_choice=Retina --num_epochs=20

CUDA_VISIBLE_DEVICES=0 python main.py --model_type=UNet_base --data_augmentation --optimizer_type=adam --lr_scheduler=reducelronplateau --loss_function=bce_loss --batch_size=8 --data_choice=Retina --num_epochs=20
CUDA_VISIBLE_DEVICES=0 python main.py --model_type=UNet_base --data_augmentation --optimizer_type=adam --lr_scheduler=reducelronplateau --loss_function=dice_loss --batch_size=8 --data_choice=Retina --num_epochs=20

CUDA_VISIBLE_DEVICES=0 python main.py --model_type=UNet --data_augmentation --optimizer_type=adam --lr_scheduler=reducelronplateau --loss_function=bce_loss --batch_size=8 --data_choice=Retina --num_epochs=20
CUDA_VISIBLE_DEVICES=0 python main.py --model_type=UNet --data_augmentation --optimizer_type=adam --lr_scheduler=reducelronplateau --loss_function=dice_loss --batch_size=8 --data_choice=Retina --num_epochs=20