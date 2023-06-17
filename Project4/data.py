import os
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import DataLoader


data_path = '/dtu/datasets1/02514/data_wastedetection'
annotation_file = os.path.join(data_path, 'annotations.json')

class WasteSet(dset.CocoDetection):
    def __init__(
            self,
            root: str,
            annFile: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            transforms: Optional[Callable] = None,
            supercategories: Optional[bool] = True
        ) -> None:
        super().__init__(root, annFile, transform, target_transform, transforms)
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.supercategories = supercategories

        if not self.supercategories:
            self.cat_names = tuple([cat['name'] for cat in self.coco.cats.values()])
        else:
            self.cat_to_supcat = {} # Category to supercategory.
            self.supcat_to_cat = {} # Supercategory to category.
            self.cat_names = list() # Category names.
            
            supcat_per_id = [cat['supercategory'] for cat in self.coco.cats.values()]
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
            self.cat_names = tuple(self.cat_names)
          
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)
        
        if self.supercategories:
            for ann in target:
                catid = ann['category_id']
                ann['category_id'] = self.cat_to_supcat[catid]

        if self.transforms is not None:
            image, target = self.transforms(image, target)
        
        if self.transform is not None:
            image = self.transform(image)
        
        if self.target_transform is not None:
            target = self.target_transform(image)
        
        return image, target


def get_waste(batch_size: int, num_workers: int = 8, data_augmentation: bool = True, supercategories: bool = True):
    train_dataset = WasteSet(data_path, annotation_file, supercategories=supercategories)
    val_dataset = WasteSet(data_path, annotation_file, supercategories=supercategories)
    test_dataset = WasteSet(data_path, annotation_file, supercategories=supercategories)
    
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
