import json
import os

import CNN
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.utils as utils
from data import get_dataloaders
from EarlyStopping import EarlyStopper
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
from tqdm import tqdm

# Hyperparameters
# TODO: Get hyperparameters in argparser or similar.
# Model
model_type = 'ResNet'
num_res_blocks = 9 # Only relevant for ResNet.
dropout = 0

# Data
batch_size = 64
data_augmentation = True
num_workers = 8

# Optimization/training
optim_type = 'adam' # adam or sgd
lrscheduler_type = 'reducelronplateau' # reducelronplateau or expdecay
early_stopping = True
early_stopping_patience = 10 # Only relevant for if early_stopping = True
lr = 1e-3
num_epochs = 100 

# Paths
save_dir = f'models/{model_type.lower()}'
if model_type.lower() == 'resnet':
    save_dir = save_dir + str(num_res_blocks)
data_dir = 'exercises/data'

if not os.path.exists('models'):
    os.mkdir('models')

if not os.path.exists(save_dir):
    os.mkdir(save_dir)
    os.mkdir(save_dir + '/checkpoints')
    os.mkdir(save_dir + '/stats')

# Check if cuda is available.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f'Using device:\t{device}')

print('Getting data...')
train_dataset, val_dataset, train_loader, val_loader = get_dataloaders(batch_size, num_workers=num_workers, data_augmentation=data_augmentation)

def train(model, optimizer, scheduler=None, earlystopper=None, num_epochs=10):

    def loss_fun(output, target):
        # NOTE: Here we assume nn.LogSoftmax output.
        return F.nll_loss(output, target)

    out_dict = {
        'train_acc': [],
        'val_acc': [],
        'train_loss': [],
        'val_loss': []
    }

    # Current learning rate
    current_lr = lr

    for epoch in range(num_epochs):
        model.train()
        #For each epoch
        train_correct = 0
        train_loss = []
        for minibatch_no, (data, target) in tqdm(enumerate(train_loader),
                                                 total=len(train_loader)):
            data, target = data.to(device), target.to(device)
            #Zero the gradients computed for each weight
            optimizer.zero_grad()
            #Forward pass your image through the network
            output = model(data)
            #Compute the loss
            loss = loss_fun(output, target)
            #Backward pass through the network
            loss.backward()
            #Update the weights
            optimizer.step()

            train_loss.append(loss.item())
            #Compute how many were correctly classified
            predicted = output.argmax(1)
            train_correct += (target == predicted).sum().cpu().item()
        #Compute the val accuracy
        val_loss = []
        val_correct = 0
        model.eval()
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            with torch.no_grad():
                output = model(data)
            val_loss.append(loss_fun(output, target).cpu().item())
            predicted = output.argmax(1)
            val_correct += (target == predicted).sum().cpu().item()

        mean_val_loss = np.mean(val_loss)

        out_dict['train_acc'].append(train_correct / len(train_dataset))
        out_dict['val_acc'].append(val_correct / len(val_dataset))
        out_dict['train_loss'].append(np.mean(train_loss))
        out_dict['val_loss'].append(mean_val_loss)
        print(
            f"Epoch: {epoch}\t Loss train: {np.mean(train_loss):.3f}\t val: {mean_val_loss:.3f}\t",
            f"Accuracy train: {out_dict['train_acc'][-1]*100:.1f}%\t val: {out_dict['val_acc'][-1]*100:.1f}%\t",
            f"Learning rate: {current_lr:.1e}",
        )

        # Learning rate scheduler step
        if scheduler is not None:
            if type(scheduler) == ReduceLROnPlateau:
                scheduler.step(mean_val_loss)
            else:
                scheduler.step()
            current_lr = scheduler._last_lr[0]
        
        # Early stopping
        if earlystopper is not None:
            if earlystopper(mean_val_loss):
                print('Training ended as the early stopping criteria was met.')
                break
    return out_dict

# Get model
in_size = (64, 64) # h, w
if model_type.lower() == 'resnet':
    model = CNN.ResNet(3, 32, in_size, num_res_blocks=num_res_blocks, dropout=dropout)
elif model_type.lower() == 'cnn_4':
    model = CNN.CNN_4(3, in_size, dropout=dropout)
model.to(device)

# Get optimizer
if optim_type.lower() == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
elif optim_type.lower() == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Get lr scheduler
if lrscheduler_type == 'reducelronplateau':
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3)
elif lrscheduler_type == 'expdecay':
    scheduler = ExponentialLR(optimizer, 2)

if early_stopping:
    earlystopper = EarlyStopper(early_stopping_patience, min_delta=0)

print(f"Training CNN model type '{model_type}' using '{optim_type.upper()}' optimization.\n'{model_type}' parameters:\n\n{model}\n\n...")

# main training loop
out_dict = train(model, optimizer, scheduler=scheduler, earlystopper=earlystopper, num_epochs=num_epochs)

# Save
idx = len(os.listdir(f'{save_dir}/checkpoints'))

if model_type.lower() == 'resnet':
    model_name = f'{model_type.lower()}{num_res_blocks}_{idx}'
else:
    model_name = f'{model_type.lower()}_{idx}'

# Checkpoint
save_path = f'{save_dir}/checkpoints/{model_name}.pt'
print(f'Saving model to:\t{save_path}')
torch.save(model.state_dict(), save_path)

# Stats
save_path = f'{save_dir}/stats/{model_name}.json'
print(f'Saving stats to:\t{save_path}')

with open(save_path, 'w') as f:
    json.dump(out_dict, f, indent=6)