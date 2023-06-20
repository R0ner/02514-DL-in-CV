import torch
import torch.nn as nn
import torchvision.models as tvm

class SimpleRCNN(nn.Module):
    def __init__(self, n_layers, n_classes):
        super(SimpleRCNN, self).__init__()
    
        self.pretrained = self.get_resnet(n_layers)
        self.new_layers = nn.Sequential(
            nn.Linear(512, n_classes)
        )
    
    def forward(self, x):
        x = self.pretrained(x)
        # Flatten the output tensor
        x = torch.flatten(x, 1)
        x = self.new_layers(x)
        return x

    def get_resnet(self, n_layers):
        supported = (18, 34, 50, 101, 152)
        assert n_layers in supported, f"'n_layers' should be one of {supported}"

        if n_layers == 18:
            model = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT) #Most up to date weights
        if n_layers == 34:
            model = tvm.resnet34(weights=tvm.ResNet34_Weights.DEFAULT) #Most up to date weights
        if n_layers == 50:
            model = tvm.resnet50(weights=tvm.ResNet50_Weights.DEFAULT) #Most up to date weights
        if n_layers == 101:
            model = tvm.resnet101(weights=tvm.ResNet101_Weights.DEFAULT) #Most up to date weights
        if n_layers == 152:
            model = tvm.resnet152(weights=tvm.ResNet152_Weights.DEFAULT) #Most up to date weights

        # Dropping the final layer of the pre-trained model
        model = nn.Sequential(*list(model.children())[:-1])  
        return model
