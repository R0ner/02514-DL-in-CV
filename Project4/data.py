import copy
import json
import os
from typing import Any, Callable, Optional, Tuple

import CocoTransforms as T
import torch
import torchvision.datasets as dset
from PIL.ImageOps import exif_transpose
from pycocotools.coco import COCO
from torch.utils.data import DataLoader
from torchvision.transforms import RandomChoice
from torchvision.transforms.functional import get_dimensions

# Paths
data_path = '/dtu/datasets1/02514/data_wastedetection'
annotation_file = os.path.join(data_path, 'annotations.json')

# Standardization is done according to training set (mean and std. for the training set)
standardize = T.Normalize([0.4960, 0.4689, 0.4142], [0.2168, 0.2089, 0.2018])
standardize_inv = T.Compose([
    T.Normalize([0, 0, 0], [1 / 0.2168, 1 / 0.2089, 1 / 0.2018]),
    T.Normalize([-0.4960, -0.4689, -0.4142], [1, 1, 1])
])


class WasteSet(dset.CocoDetection):
    """'transform' pertains to images, 'target_transform' pertains to targets, and 'transforms' pertains to both images and targets."""

    def __init__(self,
                 root: str,
                 annFile: str,
                 split: str,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 transforms: Optional[Callable] = None,
                 supercategories: Optional[bool] = True) -> None:
        super().__init__(root, annFile, transform, target_transform,
                         transforms)
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.supercategories = supercategories

        # Get ids pertiaining to split.
        with open(os.path.join(os.getcwd(), 'split.json'), 'r') as f:
            split_idx = json.loads(f.read())
        self.ids = [self.ids[i] for i in split_idx[split]]

        # Get width and height, normalize bboxes, and increment category_ids by 1 to account for background class.
        for index in range(len(self)):
            id = self.ids[index]
            target = self._load_target(id)
            for ann in target:
                im_info = self.coco.imgs[ann['image_id']]
                im_h, im_w = (im_info['height'], im_info['width'])
                ann['orig_size'] = (im_h, im_w)
                ann['size'] = (im_h, im_w)
                x, y, w, h = ann['bbox']
                ann['bbox_orig'] = [x, y, w, h]
                ann['bbox'] = [x / im_w, y / im_h, w / im_w, h / im_h]
                ann['category_id'] += 1

        if not self.supercategories:
            # Category names.
            self.cat_names = tuple(
                ['<background>'] +
                [cat['name'] for cat in self.coco.cats.values()])
        else:
            # 'Unpack' supercategories
            self.cat_to_supcat = {}  # Category to supercategory.
            self.supcat_to_cat = {}  # Supercategory to category.
            self.cat_names = list()  # Supercategory names.

            supcat_per_id = [
                cat['supercategory'] for cat in self.coco.cats.values()
            ]
            last_supcat = ''
            current_supcat_id = -1
            for id, supcat in enumerate(supcat_per_id):
                if last_supcat != supcat:
                    current_supcat_id += 1
                    last_supcat = supcat
                    self.cat_names.append(supcat)
                self.cat_to_supcat[id] = current_supcat_id

                if current_supcat_id not in self.supcat_to_cat:
                    self.supcat_to_cat[current_supcat_id] = list()
                self.supcat_to_cat[current_supcat_id].append(id)
            self.cat_names = tuple(['<background>'] + self.cat_names)

            # Set category ids to supercategories.
            for index in range(len(self)):
                id = self.ids[index]
                target = self._load_target(id)
                for ann in target:
                    catid = ann['category_id'] - 1
                    ann['category_id'] = self.cat_to_supcat[catid] + 1

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        id = self.ids[index]
        image = self._load_image(id)
        image = exif_transpose(image)
        target = copy.deepcopy(self._load_target(id))

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target


def waste_collate_fn(batch):

    im_batch = []
    target_batch = []
    for im, target in batch:
        im_batch.append(im)

        bboxes = torch.tensor([ann['bbox'] for ann in target])
        category_ids = torch.tensor([ann['category_id'] for ann in target])
        _, h, w = get_dimensions(im)
        target_batch.append({
            'bboxes_unit': bboxes,
            'bboxes': bboxes * torch.tensor([w, h, w, h]),
            'category_ids': category_ids,
            'size': (h, w)
        })

    return im_batch, target_batch


def get_waste(batch_size: int,
              num_workers: int = 8,
              data_augmentation: bool = True,
              supercategories: bool = True):
    transforms = T.Compose([T.resize(512), T.ToTensor(), standardize])
    transforms_augment = T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomApply([T.RandomResizedCrop(512, scale=(0.4, 1))],
                     p=.2),
        T.RandomApply([T.ColorJitter(.3, .3, .2, .06)], p=0.3), transforms
    ])
    if data_augmentation:
        train_dataset = WasteSet(data_path,
                                 annotation_file,
                                 'train',
                                 transforms=transforms_augment,
                                 supercategories=supercategories)
    else:
        train_dataset = WasteSet(data_path,
                                 annotation_file,
                                 'train',
                                 transforms=transforms,
                                 supercategories=supercategories)
    val_dataset = WasteSet(data_path,
                           annotation_file,
                           'val',
                           transform=transforms,
                           supercategories=supercategories)
    test_dataset = WasteSet(data_path,
                            annotation_file,
                            'test',
                            transform=transforms,
                            supercategories=supercategories)

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              collate_fn=waste_collate_fn)
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            collate_fn=waste_collate_fn)
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers,
                             collate_fn=waste_collate_fn)
    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader
