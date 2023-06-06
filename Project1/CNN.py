import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm

class ResNetBlock(nn.Module):

    def __init__(self, n_features, dropout=.5, BN=True):
        super(ResNetBlock, self).__init__()
        self.conv_block = [
            nn.Conv2d(n_features, n_features, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.BatchNorm2d(num_features=n_features), 
            nn.Conv2d(n_features, n_features, 3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=n_features)
        ]

        self.conv_block = nn.Sequential(*[m for m in self.conv_block if BN or not isinstance(m, nn.BatchNorm2d)])
        self.add_block = nn.Sequential(
            nn.ReLU(),
            nn.Dropout2d(dropout)
        )

    def forward(self, x):
        out = self.add_block(self.conv_block(x) + x)
        return out
    
class ResNet(nn.Module):

    def __init__(self, n_in, n_features, in_size, num_res_blocks=3, dropout=0.5, BN=True):
        super(ResNet, self).__init__()
        h, w = in_size

        #First conv layers needs to output the desired number of features.
        conv_layers = [
            nn.Conv2d(n_in, n_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=n_features), 
            nn.ReLU(),
            nn.Dropout2d(dropout)
        ]

        for _ in range(num_res_blocks):
            conv_layers.append(ResNetBlock(n_features, dropout=dropout, BN=BN))
        
        self.res_blocks = nn.Sequential(*[m for m in conv_layers if BN or not isinstance(m, nn.BatchNorm2d)])
        self.fc = [
            nn.Linear(h * w * n_features, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(), 
            nn.Dropout1d(dropout),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512), 
            nn.ReLU(),
            nn.Dropout1d(dropout),
            nn.Linear(512, 1)
        ]
        self.fc = nn.Sequential(*[m for m in self.fc if BN or not isinstance(m, nn.BatchNorm1d)])

    def forward(self, x):
        x = self.res_blocks(x)
        #reshape x so it becomes flat, except for the first dimension (which is the minibatch)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out

def RN18(freeze):
    print(f"Freeze set to: {freeze}")
    
    ResNet18 = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT) #Most up to date weights
    num_features = ResNet18.fc.in_features
    ResNet18.fc = nn.Linear(num_features, 1) # One logit for two classes.
    if freeze:
        for name, param in ResNet18.named_parameters():
            param.requires_grad = False
        for name, param in ResNet18.fc.named_parameters():
            param.requires_grad = True
    return ResNet18

class CNN_4(nn.Module):

    def __init__(self, n_in, in_size, dropout=0.5, BN=True):
        super(CNN_4, self).__init__()
        h, w = in_size
        
        self.convolutional = [
            nn.Conv2d(n_in, 16, 3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=8), 
            nn.ReLU(),
            nn.Dropout2d(dropout),

            nn.Conv2d(16, 16, 3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=8), 
            nn.ReLU(), 
            nn.MaxPool2d(2, stride=2),
            nn.Dropout2d(dropout),

            nn.Conv2d(16, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=16), 
            nn.ReLU(),
            nn.Dropout2d(dropout),

            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=16), 
            nn.ReLU(),
            nn.Dropout2d(dropout)
        ]

        self.fully_connected = [
            nn.Linear((h // 2) * (w // 2) * 32, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(), 
            nn.Dropout1d(dropout),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512), 
            nn.ReLU(),
            nn.Dropout1d(dropout),
            nn.Linear(512, 1)
        ]

        self.convolutional = nn.Sequential(*[m for m in self.convolutional if BN or not isinstance(m, nn.BatchNorm2d)])
        self.fully_connected = nn.Sequential(*[m for m in self.fully_connected if BN or not isinstance(m, nn.BatchNorm1d)])

    def forward(self, x):
        x = self.convolutional(x)
        #reshape x so it becomes flat, except for the first dimension (which is the minibatch)
        x = x.view(x.size(0), -1)
        x = self.fully_connected(x)
        return x
