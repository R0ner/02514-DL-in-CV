import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

from model import SimpleRCNN
from selectivesearch import SelectiveSearch
from utils import filter_and_label_proposals
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
from data import get_waste
from tqdm import tqdm
from torchvision.transforms import functional as Ft


def set_args():
    parser = argparse.ArgumentParser(description="Object Detection Training Script")
    parser.add_argument("--n_layers", type=int, default=18, help="Number of layers on in the model")
    parser.add_argument( "--n_classes", type=int, default=20, help="Number of classes in the data")
    parser.add_argument("--optimizer_type", type=str, default="adam", choices=["adam", "sgd"], help="Type of optimizer to use for training")
    parser.add_argument("--lr_scheduler",type=str,default="reducelronplateau",choices=["reducelronplateau", "expdecay"],help="Type of learning rate scheduler to use")
    parser.add_argument("--pretrained_lr", type=float, default=1e-5, help="Initial learning rate for training")
    parser.add_argument("--new_layer_lr", type=float, default=1e-3, help="Initial learning rate for training")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs to train for")
    parser.add_argument("--no_save", action="store_true", help="Whether to save the result or not")
    parser.add_argument("--batch_size", type=int, default=8, help="Number of images in each batch")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers to use for data loading")
    parser.add_argument("--data_augmentation", action="store_true", help="Whether to use data augmentation or not")
    parser.add_argument("--supercategories", action="store_true", help="Whether to use super categories")
    return parser.parse_args()
       

def get_optimizer(optim_type, model, pretrained_lr, new_layer_lr):
    if optim_type.lower() == "sgd":
        return torch.optim.SGD([
            {'params': model.pretrained.parameters(), 'lr': pretrained_lr}, 
            {'params': model.new_layers.parameters(), 'lr': new_layer_lr}
        ])
    elif optim_type.lower() == "adam":
        return torch.optim.Adam([
            {'params': model.pretrained.parameters(), 'lr': pretrained_lr}, 
            {'params': model.new_layers.parameters(), 'lr': new_layer_lr}
        ])

def get_lr_scheduler(scheduler_type, optimizer):
    if scheduler_type.lower() == "reducelronplateau":
        return ReduceLROnPlateau(optimizer, patience=9)
    elif scheduler_type.lower() == "expdecay":
        return ExponentialLR(optimizer, gamma=0.1)

def train(model, 
        optimizer,
        train_loader,
        val_loader,
        scheduler,
        num_epochs,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):

    # Selective search module
    ss = SelectiveSearch(mode='f', nkeep=100)

    # Criterions for classification loss and regression loss
    criterion =  nn.CrossEntropyLoss()
    
    model.to(device)
    out_dict = {"epoch": [], "train_loss": [], "lr": []} 
    
    for epoch in range(num_epochs):
        print("* Epoch %d/%d" % (epoch + 1, num_epochs))
        model.train()
        train_losses = []
        for ims, targets in tqdm(train_loader):

            proposals_batch = [ss((np.moveaxis(im.numpy(), 0, 2) * 255).astype(np.uint8)) for im in ims]

            proposals_batch, proposals_batch_labels = filter_and_label_proposals(proposals_batch, targets)
            boxes_batch = [np.vstack((proposal_boxes, target['bboxes'].numpy())).round().astype(int) for proposal_boxes, target in zip(proposals_batch, targets)]
            y_true = torch.tensor(np.concatenate([np.concatenate((proposal_labels, target['category_ids'].numpy())) for proposal_labels, target in zip(proposals_batch_labels, targets)]))
            

            
            optimizer.zero_grad()
            output = model(proposals_batch)
            loss = criterion(output, proposals_batch_labels)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)
        out_dict["epoch"].append(epoch)
        out_dict["train_loss"].append(avg_train_loss)
        out_dict["lr"].append(optimizer.param_groups[0]["lr"])

        print(
            f"Epoch: {epoch}, ",
            f"Learning rate: {out_dict['lr'][-1]:.1e}"
        )

    return out_dict


def main():
    args = set_args()
    print(args)
    
    # Check if cuda is available.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device:\t{device}')

    # Define model and optimizers
    model = SimpleRCNN(args.n_layers, args.n_classes)
    optimizer = get_optimizer(args.optimizer_type, model, args.pretrained_lr, args.new_layer_lr)
    lr_scheduler = get_lr_scheduler(args.lr_scheduler, optimizer)

    # Data loading
    print("Getting data...")
    _, _, _, train_loader, val_loader, test_loader = get_waste(args.batch_size,
              num_workers=args.num_workers,
              data_augmentation=args.data_augmentation,
              supercategories=args.supercategories)
    print("Done!")


    # Training
    print("Training...")
    out_dict = train(
        model, 
        optimizer,
        train_loader,
        val_loader,
        scheduler=lr_scheduler,
        num_epochs = args.num_epochs
    )
    print("Done!")



if __name__ == '__main__':
    main()