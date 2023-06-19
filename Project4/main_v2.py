import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import torchvision.transforms as transforms
import random
import torch.multiprocessing as mp
from torch.cuda.amp import autocast, GradScaler

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
    parser.add_argument( "--n_classes", type=int, default=29, help="Number of classes in the data")
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

    # Resize transform
    resize = transforms.Resize((256, 256), antialias=True)

    # Selective search module
    ss = SelectiveSearch(mode='f', nkeep=100)

    # Criterions for classification loss and regression loss
    criterion =  nn.CrossEntropyLoss()
    
    model.to(device)
    out_dict = {"epoch": [], "train_loss": [], "lr": []} 
    
    # Parallel processing
    pool = mp.Pool(mp.cpu_count())

    # Mixed precision
    scaler = GradScaler()

    for epoch in range(num_epochs):
        print("* Epoch %d/%d" % (epoch + 1, num_epochs))
        model.train()
        train_losses = []
        print_loss_total = 0  # Reset every print_every
        for i, (ims, targets) in enumerate(tqdm(train_loader)):

            #proposals_batch = [ss((np.moveaxis(im.numpy(), 0, 2) * 255).astype(np.uint8)) for im in ims]
            proposals_batch = pool.map(ss, [(np.moveaxis(im.numpy(), 0, 2) * 255).astype(np.uint8) for im in ims]) # Multiprocessing for Selective search

            proposals_batch, proposals_batch_labels = filter_and_label_proposals(proposals_batch, targets)
            boxes_batch = [np.vstack((proposal_boxes, target['bboxes'].numpy())).round().astype(int) 
                                                 for proposal_boxes, target in zip(proposals_batch, targets)]
            
            y_true = torch.tensor(np.concatenate([np.concatenate((proposal_labels, target['category_ids'].numpy())) 
                                                  for proposal_labels, target in zip(proposals_batch_labels, targets)]))
            
            #X = [resize.forward(im[:, y:y+h, x:x+w]) for im, boxes in zip(ims, boxes_batch) for x, y, w, h in boxes]
            #random.shuffle(X)
            #X = torch.stack(X).to(device)
            X = torch.stack([resize.forward(im[:, y:y+h, x:x+w]) for im, boxes in zip(ims, boxes_batch) for x, y, w, h in boxes])

            X = X.to(device)
            y_true = y_true.to(device)
            y_true = y_true.long()
            
            optimizer.zero_grad()
            with autocast():
                output = model(X)
                loss = criterion(output, y_true)

            #loss.backward()
            #optimizer.step()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_losses.append(loss.item())

            print_loss_total += loss.item()
            if (i + 1) % 10 == 0:
                print_loss_avg = print_loss_total / 10
                print_loss_total = 0
                print(f"[{i+1}] Average loss: {print_loss_avg:.2f}")

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