import glob
import os

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.transforms.functional as FT
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
import numpy as np
from pathlib import Path
import random

# Standardization is done according to training set (mean and std. for the training set)
standardize_retina = transforms.Normalize([0.4723, 0.3084, 0.1978],
                                          [0.3128, 0.2007, 0.1210])
standardize_retina_inv = transforms.Compose([
    transforms.Normalize([0, 0, 0], [1 / 0.3128, 1 / 0.2007, 1 / 0.1210]),
    transforms.Normalize([-0.4723, -0.3084, -0.1978], [1, 1, 1])
])

# TODO: Not done for the skinlesion data set...
# Standardization is done according to training set (mean and std. for the training set)
standardize_skinlesion = transforms.Normalize([0.7508, 0.5745, 0.4853],
                                              [0.1632, 0.1582, 0.1579])
standardize_skinlesion_inv = transforms.Compose([
    transforms.Normalize([0, 0, 0], [1 / 0.1632, 1 /0.1582,  1 / 0.1579]),
    transforms.Normalize([-0.7508, -0.5745, -0.4853], [1, 1, 1])
])


class RetinaSet(torch.utils.data.Dataset):
    """Dataset class for the Retina dataset"""

    def __init__(self,
                 split,
                 transform=None,
                 transform_shared=None,
                 data_path='/dtu/datasets1/02514/DRIVE'):

        # Hardcode splits :)
        self.val_indices = [0, 7, 15, 17]
        self.train_indices = [
            idx for idx in range(20) if idx not in self.val_indices
        ]

        # Initialization
        self.split = split  # train, val, or test
        self.test = self.split == 'test'
        self.data_path = data_path
        self.transform = transform
        self.transform_shared = transform_shared
        self.data_path = os.path.join(data_path,
                                      'training' if not self.test else 'test')
        self.image_paths = sorted(
            glob.glob(os.path.join(self.data_path, 'images/*.tif')))
        if not self.test:
            indices = self.train_indices if self.split == 'train' else self.val_indices

            self.label_paths = sorted(
                glob.glob(self.data_path + '/1st_manual/*.gif'))

            self.image_paths = list(
                map(lambda idx: self.image_paths[idx], indices))
            self.label_paths = list(
                map(lambda idx: self.label_paths[idx], indices))

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


class SkinLesion(torch.utils.data.Dataset):

    def __init__(self,
                 train,
                 transform,
                 data_indices,
                 data_path='/dtu/datasets1/02514/PH2_Dataset_images'):
        'Initialization'
        self.image_paths = []
        self.label_paths = []
        self.data_path = data_path
        self.transform = transform
        
        for p in Path(data_path).glob('IMD*'):
            self.p_id = p.name

            for f1 in Path(os.path.join(
                    self.data_path, self.p_id)).glob('*_Dermoscopic_Image'):
                self.img_path = sorted(glob.glob(os.path.join(
                    f1, '*.bmp')))  # base image
                self.image_paths.extend(self.img_path)

            for f2 in Path(os.path.join(self.data_path,
                                        self.p_id)).glob('*_lesion'):
                self.lbl_path = sorted(glob.glob(os.path.join(
                    f2, '*.bmp')))  # mask
                self.label_paths.extend(self.lbl_path)

            for f3 in Path(os.path.join(self.data_path,
                                        self.p_id)).glob('*_roi'):
                self.roi_paths = sorted(glob.glob(os.path.join(
                    f3, '*.bmp')))  # Multiclass segmentation if time


        ### Split is 80/10/10 percent. Randomized indexes in get_skinlesion.
        if train == 'train':
            self.image_paths = [
                self.image_paths[i] for i in data_indices[0:160]
            ]
            self.label_paths = [
                self.label_paths[i] for i in data_indices[0:160]
            ]
        elif train == 'val':
            self.image_paths = [
                self.image_paths[i] for i in data_indices[160:180]
            ]
            self.label_paths = [
                self.label_paths[i] for i in data_indices[160:180]
            ]
        elif train == 'test':
            self.image_paths = [
                self.image_paths[i] for i in data_indices[180:200]
            ]
            self.label_paths = [
                self.label_paths[i] for i in data_indices[180:200]
            ]

    def __len__(self):
        'Returns the total number of samples'
        return len(self.image_paths)

    def __getitem__(self, idx):
        'Generates one sample of data'
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]

        image = Image.open(image_path)
        label = Image.open(label_path)
        Y = FT.to_tensor(label)
        X = self.transform(image)
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

    def __init__(self,
                 degrees,
                 interpolation=InterpolationMode.NEAREST,
                 expand=False,
                 center=None,
                 fill=0):
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

        return [
            FT.rotate(img, angle, self.interpolation, self.expand, self.center,
                      fill) for img in imgs
        ]


def get_retina(batch_size: int,
               num_workers: int = 8,
               data_augmentation: bool = True):

    # Image transforms
    transform = transforms.Compose([transforms.ToTensor(), standardize_retina])
    transform_augment = transforms.Compose([
        transforms.RandomApply([transforms.ColorJitter(.3, .5, .1, .02)],
                               p=.9),
        transform
    ])

    # Shared transforms between label and image.
    transform_augment_shared = transforms.Compose([
        SegRandomHorizontalFlip(p=0.5),
        transforms.RandomApply([SegRandomRotation(180)], p=0.75)
    ])

    if data_augmentation:
        train_dataset = RetinaSet('train', transform_augment, transform_augment_shared)
    else:
        train_dataset = RetinaSet('train', transform)

    val_dataset = RetinaSet('val', transform)
    test_dataset = RetinaSet('test', transform)  # No labels!

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


def get_skinlesion(batch_size: int,
                   num_workers: int = 8,
                   data_augmentation: bool = True):

    transform = transforms.Compose(
        [transforms.ToTensor(), standardize_skinlesion])

    # Shared transforms
    transform_shared = transforms.Compose([
        SegRandomHorizontalFlip(p=0.5),
        transforms.RandomApply([SegRandomRotation(180)], p=0.75),
        # transforms.RandomApply([transforms.ColorJitter(.2, .2, .1, .05)], p=0.1),
        # transform
    ])
    random.seed(42)
    random_idxs = random.sample(range(200), 200)
    if data_augmentation:
        train_dataset = SkinLesion('train', transform, data_indices=random_idxs)
    else:
        train_dataset = SkinLesion('train', transform, data_indices=random_idxs)

    val_dataset = SkinLesion('val', transform, data_indices=random_idxs)
    test_dataset = SkinLesion('test', transform, data_indices=random_idxs)
    

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)
    val_loader = DataLoader(val_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers)

    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader