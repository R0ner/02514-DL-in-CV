from typing import Tuple, Union
import torch
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
from torchvision.transforms.functional import InterpolationMode


class Compose(transforms.Compose):

    def __call__(self, img, target=None):
        for t in self.transforms:
            img, target = t(img, target)
        return img, target

class RandomApply(transforms.RandomApply):

    def forward(self, img, target):
        if self.p < torch.rand(1):
            return img, target
        for t in self.transforms:
            img, target = t(img, target)
        return img, target

class ToTensor(transforms.ToTensor):

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
    
    def forward(self, img, target):
        """
        Args:
            img (PIL Image or Tensor): Input image.
            target: COCO annotation.

        Returns:
            PIL Image or Tensor: Color jittered image.
        """
        return super().forward(img), target

class Resize(transforms.Resize):
    
    def forward(self, img, target):
        """
        Args:
            img (PIL Image or Tensor): Image to be scaled.

        Returns:
            PIL Image or Tensor: Rescaled image.
        """
        img = F.resize(img, self.size, self.interpolation, self.max_size, self.antialias)
        if not isinstance(img, torch.Tensor):
            w_resize, h_resize = img.size
        else:
            C, h_resize, w_resize = img.shape
        # print(h_resize, w_resize)
        for ann in target:
            im_h, im_w = ann['size']
            ann['size'] = (h_resize, w_resize)
            
            # x, y, w, h = ann['bbox']
            # h_ratio, w_ratio = h_resize / im_h, w_resize / im_w
            # ann['bbox'] = [x * w_ratio, y * h_ratio, w * w_ratio, h * h_ratio]
            
        return img, target