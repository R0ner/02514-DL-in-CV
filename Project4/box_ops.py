import numpy as np


def nms(boxes, scores=None, iou_threshold=.5):
    """https://github.com/amusi/Non-Maximum-Suppression/blob/master/nms.py"""
    # If no bounding boxes, return empty list
    if len(boxes) == 0:
        return [], []

    # Bounding boxes
    boxes = np.array(boxes)

    # coordinates of bounding boxes
    start_x = boxes[:, 0]
    start_y = boxes[:, 1]
    end_x = boxes[:, 2]
    end_y = boxes[:, 3]

    # Confidence scores of bounding boxes
    if scores is None:
        scores = np.ones(boxes.shape[0])
    else:
        scores = np.array(scores).ravel()

    # Picked bounding boxes
    picked_boxes = []
    picked_scores = []

    # Compute areas of bounding boxes
    areas = (end_x - start_x + 1) * (end_y - start_y + 1)

    # Sort by confidence score of bounding boxes
    order = np.argsort(scores)

    # Iterate bounding boxes
    while order.size > 0:
        # The index of largest confidence score
        index = order[-1]

        # Pick the bounding box with largest confidence score
        picked_boxes.append(boxes[index].tolist())
        picked_scores.append(scores[index].item())

        # Compute ordinates of intersection-over-union(IOU)
        x1 = np.maximum(start_x[index], start_x[order[:-1]])
        x2 = np.minimum(end_x[index], end_x[order[:-1]])
        y1 = np.maximum(start_y[index], start_y[order[:-1]])
        y2 = np.minimum(end_y[index], end_y[order[:-1]])

        # Compute areas of intersection-over-union
        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)
        intersection = w * h

        # Compute the ratio between intersection and union
        ratio = intersection / (areas[index] + areas[order[:-1]] -
                                intersection)

        remaining = np.where(ratio < iou_threshold)
        order = order[remaining]

    return picked_boxes, picked_scores


def compute_iou(b1, b2):
    """
    Args:
        b1 and b2: bounding boxes with(x, y, w, h)
    Returns:
        IoU between b1 and b2 (float)
    """
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    # compute the coordinates of the intersection rectangle
    ix1 = max(x1, x2)
    iy1 = max(y1, y2)
    ix2 = min(x1 + w1, x2 + w2)
    iy2 = min(y1 + h1, y2 + h2)

    # If intersection is 0 return 0.
    if ix2 - ix1 <= 0 or iy2 - iy1 <= 0:
        return 0.0

    # Intesection area.
    intersection = (ix2 - ix1) * (iy2 - iy1)

    # Intersection over union
    return intersection / (w1 * h1 + w2 * h2 - intersection)


def compute_ious(box, boxes):
    """Computes the IoU between 1 bbox 'box' (1, 4) and all bboxes in 'boxes' (n, 4)"""
    x1, y1, w1, h1 = box
    x2, y2, w2, h2 = boxes.T  # Assuming bboxes is a (n, 4) numpy array

    # compute the coordinates of the intersection rectangle
    ix1 = np.maximum(x1, x2)
    iy1 = np.maximum(y1, y2)
    ix2 = np.minimum(x1 + w1, x2 + w2)
    iy2 = np.minimum(y1 + h1, y2 + h2)

    # Intesection area.
    intersection = (ix2 - ix1) * (iy2 - iy1)
    
    # Intersection over union
    ious = intersection / (w1 * h1 + w2 * h2 - intersection + 1e-7)
    
    # If intersection is 0 return 0.
    mask = ((ix2 - ix1) <= 0) | ((iy2 - iy1) <= 0)
    
    ious[mask] = 0

    return ious 