import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152

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
        x = x.view(x.size(0), -1)
        x = self.new_layers(x)
        return x

    def get_resnet(self, n_layers):
        supported = (18, 34, 50, 101, 152)
        assert n_layers in supported, f"'n_layers' should be one of {supported}"

        if n_layers == 18:
            model = resnet18(pretrained=True) 
        elif n_layers == 34:
            model = resnet34(pretrained=True)
        elif n_layers == 50:
            model = resnet50(pretrained=True) 
        elif n_layers == 101:
            model = resnet101(pretrained=True)
        elif n_layers == 152:
            model = resnet152(pretrained=True)

        # Dropping the final layer of the pre-trained model
        model = nn.Sequential(*list(model.children())[:-1])  
        return model
