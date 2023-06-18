from typing import Tuple, Union
import torch
import torchvision.transforms as transforms
from torchvision.transforms import functional as F


class Compose(transforms.Compose):
    def __init__(self, transforms):
        super().__init__(transforms)
    
    def __call__(self, img, target=None):
        for t in self.transforms:
            img, target = t(img, target)
        return img, target

class RandomApply(transforms.RandomApply):
    def __init__(self, transforms, p=0.5):
        super().__init__(transforms, p)
    
    def forward(self, img, target):
        if self.p < torch.rand(1):
            return img, target
        for t in self.transforms:
            img, target = t(img, target)
        return img, target

class ToTensor(transforms.ToTensor):
    def __init__(self) -> None:
        super().__init__()
    
    def __call__(self, pic, target=None):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.
            target: COCO annotation
        Returns:
            Tensor: Converted image.
            target: unchanged
        """
        return super().__call__(pic), target

class Normalize(transforms.Normalize):
    def __init__(self, mean, std, inplace=False):
        super().__init__(mean, std, inplace)
    
    def forward(self, tensor, target=None):
        """
        Args:
            tensor (Tensor): Tensor image to be normalized.
            target: COCO annotation
        Returns:
            Tensor: Normalized Tensor image.
            target: unchanged
        """
        return super().forward(tensor), target

class RandomHorizontalFlip(transforms.RandomHorizontalFlip):
    def __init__(self, p=0.5):
        super().__init__(p)
    
    def forward(self, img, target):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.
            target: COCO annotation
        Returns:
            PIL Image or Tensor: Randomly flipped image.
            target: COCO annotation
        """
        if torch.rand(1) < self.p:
            for ann in target:
                x, y, w, h = ann['bbox']
                ann['bbox'] = [1 - x - w, y, w, h]
            return F.hflip(img), target
        return img, target

class ColorJitter(transforms.ColorJitter):
    def __init__(self, brightness = 0.0, contrast = 0.0, saturation = 0.0, hue = 0.0):
        super().__init__(brightness, contrast, saturation, hue)
    
    def forward(self, img, target):
        """
        Args:
            img (PIL Image or Tensor): Input image.
            target: COCO annotation.

        Returns:
            PIL Image or Tensor: Color jittered image.
        """
        return super().forward(img), target