import json
import os
import argparse
import numpy as np
import torch
from model import CNN, UNet, UNet_base
from data import get_skinlesion, get_retina
from loss import bce_loss, dice_loss, focal_loss, bce_total_variation
from eval_metrics import (
    jaccard,
    dice_eval,
    accuracy,
    precision,
    recall,
    mean_score_and_conf_interval,
)
from EarlyStopping import EarlyStopper
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
from tqdm import tqdm


def set_args():
    parser = argparse.ArgumentParser(description="Segmentation Training Script")
    parser.add_argument(
        "--n_features", type=int, default=64, help="Number of features in the model"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="CNN",
        choices=["CNN", "UNet", "UNet_base"],
        help="Type of model to use for training",
    )
    parser.add_argument(
        "--data_augmentation",
        action="store_true",
        help="Whether to use data augmentation or not",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Size of each batch during training"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of workers to use for data loading",
    )
    parser.add_argument(
        "--optimizer_type",
        type=str,
        default="adam",
        choices=["adam", "sgd"],
        help="Type of optimizer to use for training",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="reducelronplateau",
        choices=["reducelronplateau", "expdecay"],
        help="Type of learning rate scheduler to use",
    )
    parser.add_argument(
        "--early_stopping",
        action="store_true",
        help="Whether to use early stopping or not",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=18,
        help="Number of patience epochs for early stopping",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Initial learning rate for training"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=100, help="Number of epochs to train for"
    )
    parser.add_argument(
        "--loss_function",
        type=str,
        default="bce_loss",
        choices=["bce_loss", "dice_loss", "focal_loss", "bce_total_variation"],
        help="Type of loss function to use for training",
    )
    parser.add_argument(
        "--data_choice",
        type=str,
        default="SkinLesion",
        choices=["SkinLesion", "Retina"],
        help="Which dataset to use for training",
    )
    parser.add_argument(
        "--no_save",
        action="store_true",
        help="Whether to save the result or not",
    )
    return parser.parse_args()


def get_model(
    model_type,
    in_channels,
    in_size,
    n_features,
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
):
    if model_type.lower() == "cnn":
        model = CNN(in_channels=in_channels, in_size=in_size, n_features=n_features)
    elif model_type.lower() == "unet_base":
        model = UNet_base(in_channels=in_channels, in_size=in_size, n_features=n_features)
    elif model_type.lower() == "unet":
        model = UNet(in_channels=in_channels, n_features=n_features)

    model.to(device)
    return model


def get_loss_func(loss_fun_type):
    if loss_fun_type.lower() == "bce_loss":
        return bce_loss
    elif loss_fun_type.lower() == "dice_loss":
        return dice_loss
    elif loss_fun_type.lower() == "focal_loss":
        return focal_loss
    elif loss_fun_type.lower() == "bce_total_variation":
        return bce_total_variation


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


def train(
    model,
    optimizer,
    loss_fun,
    train_loader,
    val_loader,
    scheduler=None,
    earlystopper=None,
    num_epochs=10,
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
) -> dict:

    model.to(device)

    out_dict = {"epoch": [], "train_loss": [], "val_loss": [], "lr": []}

    for epoch in range(num_epochs):
        print("* Epoch %d/%d" % (epoch + 1, num_epochs))
        # Train
        model.train()
        train_loss = []
        for data, target in tqdm(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fun(target, output)
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
            with torch.no_grad():
                output = model(data)
            val_loss.append(loss_fun(output, target).item())

        # Calculate average validation loss
        avg_val_loss = np.mean(val_loss)

        # Save stats
        out_dict["epoch"].append(epoch)
        out_dict["train_loss"].append(avg_train_loss)
        out_dict["val_loss"].append(avg_val_loss)
        out_dict["lr"].append(optimizer.param_groups[0]["lr"])

        print(
            f"Epoch: {epoch}, Loss - Train: {avg_train_loss:.3f}, Validation: {avg_val_loss:.3f}, "
            f"Learning rate: {out_dict['lr'][-1]:.1e}"
        )

        # Update learning rate
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(avg_val_loss)
        else:
            scheduler.step()

        # Early stopping
        if earlystopper is not None:
            if earlystopper(avg_val_loss):
                print('Training ended as the early stopping criteria was met.')
                break

    return out_dict


def evaluate_segmentation_model(
    model,
    test_loader,
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
) -> dict:
    model.to(device)
    model.eval()

    metric_functions = {
        "jaccard": jaccard,
        "dice": dice_eval,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
    }

    metrics = {"jaccard": [], "dice": [], "accuracy": [], "precision": [], "recall": []}

    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data, target = data.to(device), target.to(device)

            # Forward pass
            output = torch.sigmoid(model(data))

            # Binarize output using 0.5 as threshold
            output_binary = torch.round(output)

            # Calculate metrics
            for metric_name, metric_fn in metric_functions.items():
                score = metric_fn(target, output_binary)
                metrics[metric_name].append(score)

    # Compute mean and confidence interval for each metric
    final_metrics = {}
    for metric_name, scores in metrics.items():
        mean_score, conf_interval = mean_score_and_conf_interval(scores)
        final_metrics[metric_name] = {
            "mean": mean_score,
            "conf_interval": conf_interval,
        }

        print(
            f"{metric_name.capitalize()}:\n\tMean:\t{mean_score:.5f}\n\tC.I:\t{conf_interval:.5f}"
        )

    return final_metrics


def main():
    args = set_args()
    print(args)
    
    # Model setup
    if args.data_choice.lower() == "skinlesion":
        _in_size = (576, 752)
    elif args.data_choice.lower() == "retina":
        _in_size = (288, 288)
    
    model = get_model(
        args.model_type,
        in_channels=3,
        in_size=_in_size, #TODO: in_size is hardcoded for Retina, should probably depend on data type instead.
        n_features=args.n_features,
    )
    loss_func = get_loss_func(args.loss_function)
    optimizer = get_optimizer(args.optimizer_type, model, args.lr)
    lr_scheduler = get_lr_scheduler(args.lr_scheduler, optimizer)

    # Data loading
    print("Getting data...")
    if args.data_choice.lower() == "skinlesion":
        _, _, _, train_loader, val_loader, test_loader = get_skinlesion(
            args.batch_size,
            num_workers=args.num_workers,
            data_augmentation=args.data_augmentation,
        )
    elif args.data_choice.lower() == "retina":
        _, _, _, train_loader, val_loader, test_loader = get_retina(
            args.batch_size,
            num_workers=args.num_workers,
            data_augmentation=args.data_augmentation,
        )
        test_loader = val_loader
    print("Done!")

    # Early stopping setup
    early_stopping = (
        EarlyStopper(patience=args.early_stopping_patience)
        if args.early_stopping
        else None
    )

    # Training
    print("Training...")
    out_dict = train(
        model,
        optimizer,
        loss_func,
        train_loader,
        val_loader,
        scheduler=lr_scheduler,
        earlystopper=early_stopping,
        num_epochs=args.num_epochs,
    )

    # Evaluate model on test set
    final_metrics = evaluate_segmentation_model(model, test_loader) #TODO: Split labelled data into train, val & test

    if not args.no_save:
        # Save stats and checkpoint
        save_dir = f"models/{args.model_type.lower()}"
        if not os.path.exists("models"):
            os.mkdir("models")

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
            os.mkdir(save_dir + "/checkpoints")
            os.mkdir(save_dir + "/stats")

        # Model name for saving stats and checkpoint
        model_name = f"{args.model_type.lower()}_{args.n_features}_{args.optimizer_type}_{args.loss_function}"

        # Save used hyperparamters
        out_dict["model"] = args.model_type.lower()
        out_dict["model_name"] = model_name
        out_dict["data_augmentation"] = args.data_augmentation
        out_dict["optimizer"] = args.optimizer_type.upper()
        out_dict["loss_function"] = args.loss_function

        # Checkpoint
        save_path = f"{save_dir}/checkpoints/{model_name}.pt"
        print(f"Saving model to:\t{save_path}")
        torch.save(model.state_dict(), save_path)

        # Stats
        save_path = f"{save_dir}/stats/{model_name}.json"
        print(f"Saving training stats to:\t{save_path}")

        with open(save_path, "w") as f:
            json.dump(out_dict, f, indent=6)

        save_path = f"{save_dir}/stats/{model_name}_test_metrics.json" #TODO: can first be implemented when we have splits
        print(f"Saving test stats to:\t{save_path}")

        with open(save_path, "w") as f:
            json.dump(final_metrics, f, indent=6)


if __name__ == "__main__":
    main()
