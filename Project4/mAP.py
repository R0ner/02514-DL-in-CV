import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision


def mAP(preds, target):
    """
    preds: list of dictionaries each containing:
                boxes: torch.tensor of shape (num_boxes, 4). Each box has
                       [x,y,width,height].
                scores: torch.tensor of shape (num_boxes). A score is the
                        probability for the predicted class for a given box.
                labels: torch.tensor of shape (num_boxes). A label is the
                        predicted class label for a given box.
    target: list of dictionaries each containing:
                boxes: torch.tensor of shape (num_boxes, 4). Each box has
                       [x,y,width,height].
                labels: torch.tensor of shape (num_boxes). A label is the
                        true class label for a given box.
    :returns: the mAP estimate for the images in question!
    """
    metric = MeanAveragePrecision(box_format='xywh')
    metric.update(preds, target)
    return metric.compute()['map']

preds = [
  dict(
    boxes=torch.tensor([[258.0, 41.0, 606.0, 285.0]]),
    scores=torch.tensor([0.536]),
    labels=torch.tensor([0]),
  )
]
target = [
  dict(
    boxes=torch.tensor([[214.0, 41.0, 562.0, 285.0]]),
    labels=torch.tensor([0]),
  )
]
#metric = MeanAveragePrecision()
#metric.update(preds, target)
#from pprint import pprint
#pprint(metric.compute())
print(mAP(preds, target))