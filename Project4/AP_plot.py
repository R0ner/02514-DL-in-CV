import torch
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import pandas as pd
from map_boxes import mean_average_precision_for_boxes

#pip install Cython
#pip install map_boxes (go to init file and replace all np.str with str!)

def AP_plot(preds, targets):
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
    assert len(preds)==len(targets) #Should be equal in length
    
    pred_im_id = []
    pred_label = []
    pred_conf = []
    pred_xmin = []
    pred_xmax = []
    pred_ymin = []
    pred_ymax = []

    true_im_id = []
    true_xmin = []
    true_xmax = []
    true_ymin = []
    true_ymax = []
    true_label = []
    for i in range(len(preds)):
      pred_im_id.append(np.zeros(len(preds[i]['labels'].numpy()))+i)
      pred_conf.append(preds[i]['scores'].numpy())
      pred_xmin.append(preds[i]['boxes'][:,0].numpy())
      pred_xmax.append(preds[i]['boxes'][:,0].numpy()+preds[i]['boxes'][:,2].numpy())
      pred_ymin.append(preds[i]['boxes'][:,1].numpy())
      pred_ymax.append(preds[i]['boxes'][:,1].numpy()+preds[i]['boxes'][:,3].numpy())
      pred_label.append(preds[i]['labels'].numpy())

      prediction_on_img = np.array([preds[i]['labels'].numpy().astype(int), preds[i]['scores'].numpy(), preds[i]['boxes'][:,0].numpy(), 
      preds[i]['boxes'][:,1].numpy()+preds[i]['boxes'][:,3].numpy(), preds[i]['boxes'][:,0].numpy()+preds[i]['boxes'][:,2].numpy(),
      preds[i]['boxes'][:,1].numpy()]).T
      np.savetxt(f'Project4/images_for_AP/preds/image_{i}.txt', prediction_on_img, fmt='%f')

      true_im_id.append(np.zeros(len(targets[i]['labels'].numpy()))+i)
      true_xmin.append(targets[i]['boxes'][:,0].numpy())
      true_xmax.append(targets[i]['boxes'][:,0].numpy()+targets[i]['boxes'][:,2].numpy())
      true_ymin.append(targets[i]['boxes'][:,1].numpy())
      true_ymax.append(targets[i]['boxes'][:,1].numpy()+targets[i]['boxes'][:,3].numpy())
      true_label.append(targets[i]['labels'].numpy())

      true_img = np.array([targets[i]['labels'].numpy().astype(int), targets[i]['boxes'][:,0].numpy(), 
      targets[i]['boxes'][:,1].numpy()+targets[i]['boxes'][:,3].numpy(), targets[i]['boxes'][:,0].numpy()+targets[i]['boxes'][:,2].numpy(),
      targets[i]['boxes'][:,1].numpy()]).T
      np.savetxt(f'Project4/images_for_AP/true/image_{i}.txt', true_img, fmt='%f')
    
    print('All Done :)')

        

preds = [
  dict(
    boxes=torch.tensor([[258.0, 41.0, 606.0, 285.0], [1, 2, 3, 4], [0, 2, 3, 5]]),
    scores=torch.tensor([0.536, 0.7, 0.4]),
    labels=torch.tensor([0,1,1]),
  ),
  dict(
    boxes=torch.tensor([[258.0, 41.0, 606.0, 285.0], [1, 2, 3, 4]]),
    scores=torch.tensor([0.536, 0.7]),
    labels=torch.tensor([0,2]),
  )
]
target = [
  dict(
    boxes=torch.tensor([[214.0, 41.0, 562.0, 285.0],[1, 2, 3, 4]]),
    labels=torch.tensor([0,1]),
  ),
  dict(
    boxes=torch.tensor([[214.0, 41.0, 562.0, 285.0],[1, 2, 3, 4]]),
    labels=torch.tensor([0,2]),
  )
]

AP_plot(preds,target)