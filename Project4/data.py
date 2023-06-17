import os
from typing import Any, Callable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon, Rectangle
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import DataLoader


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
    

def show_annotation(anns, ax, supercategories=True):
    n_cats = 28 if supercategories else 60
    cmap = get_cmap(n_cats)
    for ann in anns:
        color = cmap(ann['category_id'])
        for seg in ann['segmentation']:
            poly = Polygon(np.array(seg).reshape((int(len(seg) / 2), 2)))
            p = PatchCollection([poly],
                                facecolor=color,
                                edgecolors=color,
                                linewidths=0,
                                alpha=0.4)
            ax.add_collection(p)
            p = PatchCollection([poly],
                                facecolor='none',
                                edgecolors=color,
                                linewidths=2)
            ax.add_collection(p)
        [x, y, w, h] = ann['bbox']
        rect = Rectangle((x, y),
                        w,
                        h,
                        linewidth=2,
                        edgecolor=color,
                        facecolor='none',
                        alpha=0.7,
                        linestyle='--')
        ax.add_patch(rect)


def show_cmap(names):
    n = len(names)
    cmap = get_cmap(n)
    fig, axs = plt.subplots(n, 1, figsize=(1, .25* n))
    text_kwargs = dict(ha='left', va='top', fontsize=8, color='k')
    for i, (ax, name) in enumerate(zip(axs, names)):
        t = ax.text(0, 0, name, text_kwargs)
        t.set_bbox(dict(facecolor=cmap(i)))
        ax.axis('off')


def get_cmap(n, name='hsv'):
    '''https://stackoverflow.com/questions/14720331/how-to-generate-random-colors-in-matplotlib
    Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)