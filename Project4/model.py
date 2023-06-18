import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm

def get_resnet(n_layers, n_classes):
    supported = (18, 34, 50, 101, 152)
    assert n_layers in supported, f"'n_layers' should be one of {supported}"

    if n_layers == 18:
        model = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT) #Most up to date weights
    if n_layers == 34:
        model = tvm.resnet18(weights=tvm.ResNet34_Weights.DEFAULT) #Most up to date weights
    if n_layers == 50:
        model = tvm.resnet18(weights=tvm.ResNet50_Weights.DEFAULT) #Most up to date weights
    if n_layers == 101:
        model = tvm.resnet18(weights=tvm.ResNet101_Weights.DEFAULT) #Most up to date weights
    if n_layers == 152:
        model = tvm.resnet18(weights=tvm.ResNet152_Weights.DEFAULT) #Most up to date weights
        
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, n_classes)
    
    return model
