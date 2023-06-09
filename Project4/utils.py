import matplotlib.pyplot as plt
import numpy as np
import json
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon, Rectangle
import box_ops


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
        bboxes = [box.tolist() for box in anns['bboxes_unit']]
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

def get_bbox_(target):
    """
    Gets the resized bounding box from the target(s) in the data.
    A singular target is a dict that can have multiple object bboxes inside.
    """
    out = []
    for t in range(len(target)):
        im_h, im_w = target[t]['size']
        #print(f"Target {t} has size {target[t]['size']}")
        #print("Resizing the bbox...")
        bbox = [x, y, w, h] = target[t]['bbox']
        resized_bbox = [x * im_w, y * im_h, w * im_w, h * im_h]
        out.append(resized_bbox)
    return out

def get_iou(box1, box2):
    # Extract coordinates from bounding boxes
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Calculate the coordinates of the intersection rectangle
    intersection_x1 = max(x1, x2)
    intersection_y1 = max(y1, y2)
    intersection_x2 = min(x1 + w1, x2 + w2)
    intersection_y2 = min(y1 + h1, y2 + h2)
    
    # Calculate the area of the intersection rectangle
    intersection_area = max(0, intersection_x2 - intersection_x1) * max(0, intersection_y2 - intersection_y1)
    
    # Calculate the area of each bounding box
    area_box1 = w1 * h1
    area_box2 = w2 * h2
    
    # Calculate the union area
    union_area = area_box1 + area_box2 - intersection_area
    
    # Calculate the IoU
    iou = intersection_area / union_area    
    return iou


def get_cmap(n, name='hsv'):
    '''https://stackoverflow.com/questions/14720331/how-to-generate-random-colors-in-matplotlib
    Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)


def filter_and_label_proposals(proposals_batch, targets, min_proposals=4):
     # TODO: validate -> filter out proposals with zero width or height right after creation
    proposals_batch_labels = []
    for i, target in enumerate(targets):
        proposals = proposals_batch[i]
        valid = (proposals[:, 2] > 2) & (proposals[:, 3] > 2)
        proposals = proposals[valid]
        if target['bboxes'].shape[0] == 0:
            proposals = proposals[np.random.choice(proposals.shape[0], size=min_proposals, replace=False)]
            proposal_labels = np.array(min_proposals * [0])
        else:
            proposal_labels = np.zeros(proposals.shape[0])
            # proposals_unit = proposals / np.array([w, h, w, h])
            ious = np.stack([box_ops.compute_ious(box.numpy(), proposals) for box in target['bboxes']])
            mask = (ious > .5).any(axis=0)
            ious_filter = ious[:, mask]
            proposal_labels[mask] = target['category_ids'].numpy()[ious_filter.argmax(0)]
            
            # Include all positives and 3/4 parts background.
            include = np.where(proposal_labels != 0)[0]
            include = np.concatenate((include, np.where(proposal_labels == 0)[0][:max(3 * include.size, min_proposals)]))
            
            proposals = proposals[include]
            proposal_labels = proposal_labels[include]

        proposals_batch[i] = proposals
        proposals_batch_labels.append(proposal_labels)
    
    return proposals_batch, proposals_batch_labels