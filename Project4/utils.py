import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon, Rectangle


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