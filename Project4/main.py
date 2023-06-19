import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model import get_resnet
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
from data import get_waste
from tqdm import tqdm


def set_args():
    parser = argparse.ArgumentParser(description="Object Detection Training Script")
    parser.add_argument("--n_layers", type=int, default=18, help="Number of layers on in the model")
    parser.add_argument( "--n_classes", type=int, default=20, help="Number of classes in the data")
    parser.add_argument("--optimizer_type", type=str, default="adam", choices=["adam", "sgd"], help="Type of optimizer to use for training")
    parser.add_argument("--lr_scheduler",type=str,default="reducelronplateau",choices=["reducelronplateau", "expdecay"],help="Type of learning rate scheduler to use")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate for training")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs to train for")
    parser.add_argument("--no_save", action="store_true", help="Whether to save the result or not")
    parser.add_argument("--batch_size", type=int, default=8, help="Number of images in each batch")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers to use for data loading")
    parser.add_argument("--data_augmentation", action="store_true", help="Whether to use data augmentation or not")
    parser.add_argument("--supercategories", action="store_true", help="Whether to use super categories")
    return parser.parse_args()
       

def get_optimizer(optim_type, model, lr):
    if optim_type.lower() == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr)
    elif optim_type.lower() == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr)

def get_lr_scheduler(scheduler_type, optimizer):
    if scheduler_type.lower() == "reducelronplateau":
        return ReduceLROnPlateau(optimizer, patience=9)
    elif scheduler_type.lower() == "expdecay":
        return ExponentialLR(optimizer, gamma=0.1)
    
def loss_fun(cls_scores, bbox_deltas, labels, gt_bbox):
    # Classification loss
    cls_loss = F.cross_entropy(cls_scores, labels, reduction='mean')

    # Bounding box regression loss
    bbox_loss = F.smooth_l1_loss(bbox_deltas, gt_bbox, reduction='mean')

    # Total loss is a sum of classification and regression loss
    total_loss = cls_loss + bbox_loss
    return total_loss

def train(model, 
        optimizer,
        train_loader,
        val_loader,
        scheduler,
        num_epochs,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    
    model.to(device)
    out_dict = {"epoch": [], "train_loss": [], "val_loss": [], "lr": []} #TODO: what metrics should we track?
    
    for epoch in range(num_epochs):
        print("* Epoch %d/%d" % (epoch + 1, num_epochs))
        # Train
        model.train()
        train_loss = []
        for data, target in tqdm(train_loader):
            data, target = data.to(device), target.to(device)
            labels = torch.tensor([ann['category_id'] for ann in target]) # Not sure on this part
            gt_boxes = torch.tensor([ann['bbox'] for ann in target]) # Not sure on this part
            optimizer.zero_grad()
            cls_scores, bbox_deltas = model(data)
            loss = loss_fun(cls_scores, bbox_deltas, labels, gt_boxes)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        
        # Calculate average training loss
        avg_train_loss = np.mean(train_loss)

        # Validate
        model.eval()
        val_loss = []
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            labels = torch.tensor([ann['category_id'] for ann in target]) # Not sure on this part
            gt_boxes = torch.tensor([ann['bbox'] for ann in target]) # Not sure on this part
            with torch.no_grad():
                cls_scores, bbox_deltas = model(data)
            val_loss.append(loss_fun(cls_scores, bbox_deltas, labels, gt_boxes).item())
        
        # Calculate average validation loss
        avg_val_loss = np.mean(val_loss)

        # Save stats
        out_dict["epoch"].append(epoch)
        out_dict["train_loss"].append(avg_train_loss)
        out_dict["val_loss"].append(avg_val_loss)
        out_dict["lr"].append(optimizer.param_groups[0]["lr"])

        print(
            f"Epoch: {epoch}, "
            f"Loss - Train: {avg_train_loss:.3f}, Validation: {avg_val_loss:.3f}, "
            f"Learning rate: {out_dict['lr'][-1]:.1e}"
        )

        # Update learning rate
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(avg_val_loss)
        else:
            scheduler.step()

    return out_dict


def main():
    args = set_args()
    print(args)
    
    # Check if cuda is available.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device:\t{device}')

    # Define model and optimizers
    model = get_resnet(args.n_layers, args.n_classes)
    optimizer = get_optimizer(args.optimizer_type, model, args.lr)
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

    # Evaluate model on test set



if __name__ == '__main__':
    main()