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
            target: COCO annotation

        Returns:
            PIL Image or Tensor: Rescaled image.
            target: COCO annotation
        """
        img = F.resize(img, self.size, self.interpolation, self.max_size, self.antialias)
        
        _, h_resize, w_resize = F.get_dimensions(img)

        for ann in target:
            ann['size'] = (h_resize, w_resize)
        
        return img, target

class RandomResizedCrop(transforms.RandomResizedCrop):

    def forward(self, img, target):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped and resized.
            target: COCO annotation

        Returns:
            PIL Image or Tensor: Randomly cropped and resized image.
            target: COCO annotation
        """
        i, j, _h, _w = self.get_params(img, self.scale, self.ratio)
        
        img = F.resized_crop(img, i, j, _h, _w, self.size, self.interpolation, antialias=self.antialias)

        _, h_resize, w_resize = F.get_dimensions(img)
        keep = []
        for idx, ann in enumerate(target):
            (im_h, im_w) = ann['size']
            ann['size'] = (h_resize, w_resize)
            im_x0, im_y0, w_ratio, h_ratio = j / im_w, i / im_h, im_w / _w, im_h / _h 

            x0, y0, w, h = ann['bbox']
            x1, y1 = x0 + w, y0 + h

            x0, x1, y0, y1 = (x0 - im_x0) * w_ratio, (x1 - im_x0) * w_ratio, (y0 - im_y0) * h_ratio, (y1 - im_y0) * h_ratio
            if x1 < 0 or y1 < 0 or x0 > 1 or y0 > 1:
                continue
            else:
                x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, 1), min(y1, 1)
                ann['bbox'] = [x0, y0, x1 - x0, y1 - y0]
                keep.append(idx)
        
        # Discard 'out of bounds' boxes.
        target = [target[idx] for idx in keep]

        return img, target

transforms.RandomChoice