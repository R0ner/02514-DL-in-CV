import matplotlib.pyplot as plt
import numpy as np
import json
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon, Rectangle


def get_split():
    np.random.seed(1915)
    idx = np.arange(1500)
    np.random.shuffle(idx)

    split = {}
    # 15 % validation and test
    split['val'] = idx[:225].tolist()
    split['test'] = idx[225:450].tolist()
    split['train'] = idx[450:].tolist()
    with open('Project4/split.json', "w") as f:
        json.dump(split, f, indent=6)


def show_annotation(anns, ax, supercategories=True, names=None):
    n_cats = 29 if supercategories else 61
    cmap = get_cmap(n_cats)

    text_kwargs = dict(ha='left', va='bottom', fontsize=4, color='k')
    if isinstance(anns, dict):
        im_h, im_w = anns['size']
        bboxes = [box.tolist() for box in anns['bboxes']]
        category_ids = [category_id.item() for category_id in anns['category_ids']]
    else:
        bboxes = []
        category_ids = []
        for ann in anns:
            im_h, im_w = ann['size']
            bboxes.append(ann['bbox'])
            category_ids.append(ann['category_id'])
    for ((x,y,w,h), category_id) in zip(bboxes, category_ids):
        color = cmap(category_id)
        rect = Rectangle((x * im_w, y * im_h),
                        w * im_w,
                        h * im_h,
                        linewidth=1.5,
                        edgecolor=color,
                        facecolor='none',
                        alpha=0.7,
                        linestyle='--')
        ax.add_patch(rect)
        if names is not None:
            t = ax.text(x * im_w, y * im_h, names[category_id], text_kwargs)
            t.set_bbox(dict(facecolor=color, edgecolor=(0,0,0,0), pad=.5))


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