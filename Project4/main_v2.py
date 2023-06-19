import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

from model import SimpleRCNN
from selectivesearch import SelectiveSearch
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

def compute_ious(proposal, bboxes):
    x1, y1, w1, h1 = proposal
    x2, y2, w2, h2 = bboxes.T  # Assuming bboxes is a (n, 4) numpy array

    # Compute coordinates of intersection rectangle
    x_left = np.maximum(x1, x2)
    y_top = np.maximum(y1, y2)
    x_right = np.minimum(x1 + w1, x2 + w2)
    y_bottom = np.minimum(y1 + h1, y2 + h2)

    # Compute area of intersection
    intersection_area = np.maximum(0, x_right - x_left + 1) * np.maximum(0, y_bottom - y_top + 1)

    # Compute areas of bounding boxes
    proposal_area = w1 * h1
    bboxes_area = w2 * h2

    # Compute IoU
    ious = intersection_area / (proposal_area + bboxes_area - intersection_area)

    return ious

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
    out_dict = {"epoch": [], "train_loss_cls": [], "train_loss_reg": [], "lr": []} #TODO: what metrics should we track?
    
    for epoch in range(num_epochs):
        print("* Epoch %d/%d" % (epoch + 1, num_epochs))
        # Train
        model.train()
        train_loss = []
        for data, target in tqdm(train_loader):
            data = data.to(device)
            
            target_bboxes = torch.tensor([ann['category_id'] for ann in target], device=device)
            target_labels = torch.tensor([ann['bbox'] for ann in target], device=device)

            # Get region proposals and preprocess them
            boxes = ss((np.moveaxis(data.numpy(), 0, 2) * 255).astype(np.uint8))
            
            # Train model for each region proposals
            batch_losses = []
            for box in boxes:
                
                x_start = max(0, box[0])
                y_start = max(0, box[1])
                x_end = min(data.shape[2], box[0] + box[2])
                y_end = min(data.shape[1], box[1] + box[3])

                cropped = data[:, y_start:y_end, x_start:x_end]
                print("Cropped shape:", cropped.shape)

                if isinstance(cropped, torch.Tensor):
                    cropped = cropped.cpu().numpy()

                proposal = cv2.resize(cropped.transpose(1,2,0), (224, 224))
                proposal = torch.from_numpy(proposal).permute(2, 0, 1).unsqueeze(0).float() / 255

                # Determine IoU with all ground truth bounding boxes
                ious = compute_ious(box, target_bboxes)

                # We define object as IoU >= 0.5, and background as IoU < 0.5
                if ious.max() < 0.5:
                    target_label = torch.tensor([0]) # Background
                else:
                    target_label = target_labels[ious.argmax()]
                
                # Move to GPU if available
                proposal = proposal.to(device)
                target_label = target_label.to(device)
                
                optimizer.zero_grad()
                output = model(proposal)
                loss = criterion(output, target_label)
                loss.backward()
                optimizer.step()
                batch_losses.append(loss.item())

            train_loss.append(np.mean(batch_losses))

        # Save stats
        avg_train_loss = sum(train_loss) / len(train_loss)
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