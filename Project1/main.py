import os

import CNN
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.utils as utils
from data import get_dataloaders
from tqdm import tqdm

# Hyperparameters
# TODO: Get hyperparameters in argparser or similar.
# Model
model_type = 'ResNet'

# Data
batch_size = 64
data_augmentation = True
num_workers = 8

# Optimization/training
optim_type = 'adam' # adam or sgd
lr = 1e-3
num_epochs = 10  # TODO: Implement early stopping.

# Paths
save_path = f'models/{model_type}.pt'
data_dir = 'exercises/data'

if not os.path.exists('models'):
    os.mkdir('models')

# Check if cuda is available.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f'Using device:\t{device}')

print('Getting data...')
train_dataset, val_dataset, train_loader, val_loader = get_dataloaders(batch_size, num_workers=num_workers, data_augmentation=data_augmentation)

def train(model, optimizer, num_epochs=10):

    def loss_fun(output, target):
        # NOTE: Here we assume nn.LogSoftmax output.
        return F.nll_loss(output, target)

    out_dict = {
        'train_acc': [],
        'val_acc': [],
        'train_loss': [],
        'val_loss': []
    }

    for epoch in tqdm(range(num_epochs), unit='epoch'):
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
        #Comput the val accuracy
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
        out_dict['train_acc'].append(train_correct / len(train_dataset))
        out_dict['val_acc'].append(val_correct / len(val_dataset))
        out_dict['train_loss'].append(np.mean(train_loss))
        out_dict['val_loss'].append(np.mean(val_loss))
        print(
            f"Loss train: {np.mean(train_loss):.3f}\t val: {np.mean(val_loss):.3f}\t",
            f"Accuracy train: {out_dict['train_acc'][-1]*100:.1f}%\t val: {out_dict['val_acc'][-1]*100:.1f}%"
        )
    return out_dict

# Get model
in_size = (64, 64) # h, w
if model_type.lower() == 'resnet':
    model = CNN.ResNet(3, 16, in_size)
elif model_type.lower() == 'cnn_4':
    model = CNN.CNN_4(3, in_size, dropout=0.5)

# Get optimizer
if optim_type.lower() == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
elif optim_type.lower() == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

print(f"Training CNN model type '{model_type}' using '{optim_type.upper()}' optimization.\n'{model_type}' parameters:\n\n{model}\n\n...")

# main training loop
out_dict = train(model, optimizer)

# Save
print(f'Saving model to:\t{save_path}')
torch.save(model.state_dict(), save_path)