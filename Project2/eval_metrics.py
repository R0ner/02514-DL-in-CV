import torch
from sklearn.metrics import precision_score, recall_score
import numpy as np

def image_to_evaltensor(image):
    """
    Takes a torch image and returns a flattened torch tensor
    """
    return image.view(-1)

def jaccard(y_true, y_pred):
    """
    All evaluation metrics assume input to be flattened torch tensors of 
    ground truth labels (y_true) and predicted labels (y_pred).
    They then return the evaluation metric in question.
    """
    y_true = image_to_evaltensor(y_true)
    y_pred = image_to_evaltensor(y_pred)
    intersection = (y_true==y_pred).sum()
    union = (len(y_true)+len(y_pred)) - intersection
    return intersection / union

def dice_eval(y_true, y_pred):
    """
    All evaluation metrics assume input to be flattened torch tensors of 
    ground truth labels (y_true) and predicted labels (y_pred).
    They then return the evaluation metric in question.
    """
    y_true = image_to_evaltensor(y_true)
    y_pred = image_to_evaltensor(y_pred)
    intersection = (y_true==y_pred).sum()
    return (2*intersection) / (len(y_true)+len(y_pred))

def accuracy(y_true, y_pred):
    """
    All evaluation metrics assume input to be flattened torch tensors of 
    ground truth labels (y_true) and predicted labels (y_pred).
    They then return the evaluation metric in question.
    """
    y_true = image_to_evaltensor(y_true)
    y_pred = image_to_evaltensor(y_pred)
    intersection = (y_true==y_pred).sum()
    return intersection / len(y_true)

def precision(y_true, y_pred):
    """
    All evaluation metrics assume input to be flattened torch tensors of 
    ground truth labels (y_true) and predicted labels (y_pred).
    They then return the evaluation metric in question.
    """
    y_true = image_to_evaltensor(y_true).to_numpy()
    y_pred = image_to_evaltensor(y_pred).to_numpy()
    
    return precision_score(y_true, y_pred)

def recall(y_true, y_pred):
    """
    All evaluation metrics assume input to be flattened torch tensors of 
    ground truth labels (y_true) and predicted labels (y_pred).
    They then return the evaluation metric in question.
    """
    y_true = image_to_evaltensor(y_true).to_numpy()
    y_pred = image_to_evaltensor(y_pred).to_numpy()
    
    return recall_score(y_true, y_pred)

def mean_score_and_conf_interval(metrics: list):
    """
    Takes a list of performance metrics for each image in test set and computes
    mean (mean) and a confidence interval (conf).
    """
    mean = np.mean(metrics)
    conf = (1.96*np.std(metrics)) / np.sqrt(len(metrics))
    return mean, conf