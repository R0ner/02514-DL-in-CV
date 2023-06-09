import glob
import os

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.transforms.functional as FT
from PIL import Image
from torchvision.transforms.functional import InterpolationMode


class RetinaSet(torch.utils.data.Dataset):
    """Dataset class for the Retina dataset"""
    def __init__(self, split, transform=None, transform_shared=None, data_path='/dtu/datasets1/02514/DRIVE'):
        
        # Hardcode splits :)
        self.val_indices = [0, 7, 15, 17]
        self.train_indices = [idx for idx in range(20) if idx not in self.val_indices]

        # Initialization
        self.split = split # train, val, or test
        self.test = self.split == 'test'
        self.data_path = data_path
        self.transform = transform
        self.transform_shared = transform_shared
        self.data_path = os.path.join(data_path, 'training' if not self.test else 'test')
        self.image_paths = sorted(glob.glob(os.path.join(self.data_path, 'images/*.tif')))
        if not self.test:
            indices = self.train_indices if self.split == 'train' else self.val_indices
            
            self.label_paths = sorted(glob.glob(self.data_path + '/1st_manual/*.gif'))

            self.image_paths = list(map(lambda idx: self.image_paths[idx], indices))
            self.label_paths = list(map(lambda idx: self.label_paths[idx], indices))

    def __len__(self):
        # Returns the total number of samples
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Generates one sample of data
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        
        if not self.test:
            label_path = self.label_paths[idx]
            label = Image.open(label_path)
        else:
            label = None
        
        if self.transform_shared is not None:
            image, label = self.transform_shared([image, label])
        
        X = self.transform(image)
        Y = FT.to_tensor(label)
        return X, Y

class SegRandomHorizontalFlip(transforms.RandomHorizontalFlip):
    def __init__(self, p=0.5):
        super().__init__(p)
    
    def forward(self, imgs):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.

        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """
        if torch.rand(1) < self.p:
            return [FT.hflip(img) for img in imgs]
        return imgs

class SegRandomRotation(transforms.RandomRotation):
    def __init__(self, degrees, interpolation=InterpolationMode.NEAREST, expand=False, center=None, fill=0):
        super().__init__(degrees, interpolation, expand, center, fill)
    
    def forward(self, imgs):
        """
        Args:
            img (PIL Image or Tensor): Image to be rotated.

        Returns:
            PIL Image or Tensor: Rotated image.
        """
        fill = self.fill
        fills = []
        for img in imgs:
            channels, _, _ = FT.get_dimensions(img)
            if isinstance(img, torch.Tensor):
                if isinstance(fill, (int, float)):
                    fill = [float(fill)] * channels
                else:
                    fill = [float(f) for f in fill]
            fills.append(fill)

        angle = self.get_params(self.degrees)

        return [FT.rotate(img, angle, self.interpolation, self.expand, self.center, fill) for img in imgs]
    

def get_retina(batch_size: int, num_workers: int = 8, data_augmentation: bool = True):

    # Standardization is done according to training set (mean and std. for the training set)
    standardize = transforms.Normalize([0.4723, 0.3084, 0.1978],
                                    [0.3128, 0.2007, 0.1210])
    standardize_inv = transforms.Compose([
        transforms.Normalize([0, 0, 0], [1 / 0.3128, 1 / 0.2007, 1 / 0.1210]),
        transforms.Normalize([-0.4723, -0.3084, -0.1978], [1, 1, 1])
    ])

    
    transform = transforms.Compose([
        transforms.ToTensor(),
        standardize
    ])
    
    # Shared transforms
    transform_shared = transforms.Compose([
            SegRandomHorizontalFlip(p=0.5),
            transforms.RandomApply([SegRandomRotation(180)], p=0.75),
            # transforms.RandomApply([transforms.ColorJitter(.2, .2, .1, .05)], p=0.1),
            # transform
    ])

    if data_augmentation:
        train_dataset = RetinaSet('train', transform, transform_shared)
    else:
        train_dataset = RetinaSet('train', transform)
    
    val_dataset = RetinaSet('val', transform)
    test_dataset = RetinaSet('test', transform) # No labels!

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers)
    test_loader = DataLoader(test_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers)
    
    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader
    