import os

import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

train_dir = '/dtu/datasets1/02514/hotdog_nothotdog/train'
test_dir = '/dtu/datasets1/02514/hotdog_nothotdog/test'

assert os.path.exists(train_dir) and os.path.exists(test_dir)

# Standardization is done according to ImageNet (mean and std. for ImageNet)
standardize = transforms.Normalize([0.485, 0.456, 0.406],
                                   [0.229, 0.224, 0.225])
standardize_inv = transforms.Compose([
    transforms.Normalize([0, 0, 0], [1 / 0.229, 1 / 0.224, 1 / 0.225]),
    transforms.Normalize([-0.485, -0.456, -0.406], [1, 1, 1])
])

# Make transforms
# Shared transforms (base)
transform_base = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((64, 64), antialias=True),  # They are not all square
])

# No augmentation
transform = transforms.Compose([transform_base, standardize])

# With augmentation
# TODO: More types of augmentation?
transform_augment = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.RandomRotation(45)], p=0.4),
    transforms.RandomPerspective(distortion_scale=0.4, p=0.1),
    transforms.RandomApply([transforms.ColorJitter(.5, .5, .2, .1)], p=0.3),
    transforms.RandomApply([transforms.Grayscale(num_output_channels=3)], p=0.1),
    transforms.RandomApply([transforms.RandomResizedCrop(64, scale=(.3, 1))], p=.9),
    transform_base,
    standardize
])


def get_dataloaders(batch_size: int,
                    num_workers: int = 8,
                    data_augmentation: bool = True):
    """Get train and validation dataloaders for the hotdog dataset.
    Returns: 
        tuple(train_dataset, val_dataset, train_loader, val_loader)    torch datasets dataloaders for the train and validation set respectively"""
    if data_augmentation:
        transform_train = transform_augment
    else:
        transform_train = transform
    transform_val = transform

    train_dataset = datasets.ImageFolder(root=train_dir,
                                         transform=transform_train)
    val_dataset = datasets.ImageFolder(root=test_dir, transform=transform_val)

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers)

    return train_dataset, val_dataset, train_loader, val_loader