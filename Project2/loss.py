import torch
import torch.nn.functional as F

def bce_loss(y_real, y_pred):
    max_val = (-y_pred).clamp(min=0)
    return torch.mean(y_pred - y_pred * y_real + max_val +
                      ((-max_val).exp() + (-y_pred - max_val).exp()).log())

def dice_loss(y_real, y_pred):
    y_pred_sigmoid = F.sigmoid(y_pred)
    return 1 - (2 * y_real * y_pred_sigmoid + 1).mean() / (
        (y_real + y_pred_sigmoid).mean() + 1)

def focal_loss(y_real, y_pred, gamma=2):
    y_pred_sigmoid = F.sigmoid(y_pred)
    return -((1 - y_pred_sigmoid)**gamma * y_real * y_pred_sigmoid.log() +
             (1 - y_real) * (1 - y_pred_sigmoid).log()).mean()

def total_variation(y_pred):
    y_pred_sigmoid = F.sigmoid(y_pred)
    return (y_pred_sigmoid[..., :-1, :] - y_pred_sigmoid[..., 1:, :]).abs().mean() + (y_pred_sigmoid[..., :, :-1] -
                             y_pred_sigmoid[..., :, 1:]).abs().mean()


def bce_total_variation(y_real, y_pred):
    return bce_loss(y_real, y_pred) + 0.1 * total_variation(y_pred)