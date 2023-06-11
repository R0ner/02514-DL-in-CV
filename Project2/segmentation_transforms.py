import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as FT
from PIL import Image
from torchvision.transforms.functional import InterpolationMode


class SegResize(transforms.Resize):

    def __init__(self, size, interpolation=InterpolationMode.BILINEAR, max_size=None, antialias="warn"):
        super().__init__(size, interpolation, max_size, antialias)
    
    def forward(self, imgs):
        """
        Args:
            list[imgs] list of images (PIL Image or Tensor): Images to be scaled.

        Returns:
            list[imgs] list of PIL Image or Tensor: Rescaled images.
        """
        imgs_resized = []
        for img in imgs:
            imgs_resized.append(super().forward(img))
        return imgs_resized


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

class SegElasticTransform(transforms.ElasticTransform):
    def __init__(self, alpha=50.0, sigma=5.0, interpolation=Image.BILINEAR, fill=0):
        super().__init__(alpha, sigma, interpolation, fill)
    
    def forward(self, imgs):
        _, height, width = FT.get_dimensions(imgs[0])
        displacement = self.get_params(self.alpha, self.sigma, [height, width])

        return [FT.to_pil_image(FT.elastic_transform(FT.to_tensor(img), displacement, self.interpolation, self.fill)) for img in imgs]