import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class CNN(nn.Module):

    def __init__(self, in_channels=3, in_size=(584, 565), n_features=64):
        super().__init__()

        self.in_channels = in_channels
        self.in_size = in_size
        self.n_features = n_features

        # encoder (downsampling)
        self.enc_conv0 = nn.Conv2d(self.in_channels, self.n_features, 3, padding=1)
        self.enc_bnorm0 = nn.BatchNorm2d(self.n_features)
        self.pool0 = nn.MaxPool2d(2, 2)  # in_size -> in_size // 2
        self.enc_conv1 = nn.Conv2d(self.n_features, 2 * self.n_features, 3, padding=1)
        self.enc_bnorm1 = nn.BatchNorm2d(2 * self.n_features)
        self.pool1 = nn.MaxPool2d(2, 2)  # in_size // 2 -> in_size // 4
        self.enc_conv2 = nn.Conv2d(2 * self.n_features, 4 * self.n_features, 3, padding=1)
        self.enc_bnorm2 = nn.BatchNorm2d(4 * self.n_features)
        self.pool2 = nn.MaxPool2d(2, 2)  # in_size // 4 -> in_size // 8
        self.enc_conv3 = nn.Conv2d(4 * self.n_features, 8 * self.n_features, 3, padding=1)
        self.enc_bnorm3 = nn.BatchNorm2d(8 * self.n_features)
        self.pool3 = nn.MaxPool2d(2, 2)  # in_size // 8 -> in_size // 16

        # bottleneck
        self.bottleneck_conv = nn.Conv2d(8 * self.n_features, 16 * self.n_features, 3, padding=1)
        self.bneck_bnorm = nn.BatchNorm2d(16 * self.n_features)

        # decoder (upsampling)
        self.upsample0 = nn.Upsample((self.in_size[0] // 8, self.in_size[1] // 8)) # in_size // 16 -> in_size // 8
        self.dec_conv0 = nn.Conv2d(16 * self.n_features, 8 * self.n_features, 3, padding=1)
        self.dec_bnorm0 = nn.BatchNorm2d(8 * self.n_features)
        self.upsample1 = nn.Upsample((self.in_size[0] // 4, self.in_size[1] // 4)) # in_size // 8 -> in_size // 4
        self.dec_conv1 = nn.Conv2d(8 * self.n_features, 4 * self.n_features, 3, padding=1)
        self.dec_bnorm1 = nn.BatchNorm2d(4 * self.n_features)
        self.upsample2 = nn.Upsample((self.in_size[0] // 2, self.in_size[1] // 2)) # in_size // 4 -> in_size // 2
        self.dec_conv2 = nn.Conv2d(4 * self.n_features, 2 * self.n_features, 3, padding=1)
        self.dec_bnorm2 = nn.BatchNorm2d(2 * self.n_features)
        self.upsample3 = nn.Upsample(self.in_size) # in_size // 2 -> in_size
        self.dec_conv3 = nn.Conv2d(2 * self.n_features, 1, 3, padding=1)

    def forward(self, x):
        # encoder
        e0 = self.pool0(F.relu(self.enc_bnorm0(self.enc_conv0(x))))
        e1 = self.pool1(F.relu(self.enc_bnorm1(self.enc_conv1(e0))))
        e2 = self.pool2(F.relu(self.enc_bnorm2(self.enc_conv2(e1))))
        e3 = self.pool3(F.relu(self.enc_bnorm3(self.enc_conv3(e2))))

        # bottleneck
        b = F.relu(self.bneck_bnorm(self.bottleneck_conv(e3)))

        # decoder
        d0 = F.relu(self.dec_bnorm0(self.dec_conv0(self.upsample0(b))))
        d1 = F.relu(self.dec_bnorm1(self.dec_conv1(self.upsample1(d0))))
        d2 = F.relu(self.dec_bnorm2(self.dec_conv2(self.upsample2(d1))))
        d3 = self.dec_conv3(self.upsample3(d2))  # no activation
        return d3

class UNet_base(nn.Module):
    """Original UNet implementation with maxpooling and upsampling."""
    def __init__(self, in_channels=3, in_size=(584, 565), n_features=64):
        super().__init__()

        self.in_channels = in_channels
        self.h, self.w = in_size
        self.n_features = n_features

        # encoder (downsampling)
        self.enc_conv0 = nn.Conv2d(self.in_channels, self.n_features, 3, padding=1)
        self.pool0 = nn.MaxPool2d(2, 2) # in_size -> in_size // 2
        self.enc_conv1 = nn.Conv2d(self.n_features, self.n_features, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)  # in_size // 2 -> in_size // 4
        self.enc_conv2 = nn.Conv2d(self.n_features, self.n_features, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)  # in_size // 4 -> in_size // 8
        self.enc_conv3 = nn.Conv2d(self.n_features, self.n_features, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)  # in_size // 8 -> in_size // 16

        # bottleneck
        self.bottleneck_conv = nn.Conv2d(self.n_features, self.n_features, 3, padding=1)

        # decoder (upsampling)
        self.upsample0 = nn.Upsample((self.h // 8, self.w // 8))  # in_size // 16 -> in_size // 8
        self.dec_conv0 = nn.Conv2d(2 * self.n_features, self.n_features, 3, padding=1)
        self.upsample1 = nn.Upsample((self.h // 4, self.w // 4))  # in_size // 8 -> in_size // 4
        self.dec_conv1 = nn.Conv2d(2 * self.n_features, self.n_features, 3, padding=1)
        self.upsample2 = nn.Upsample((self.h // 2, self.w // 2))  # in_size // 4 -> in_size // 2
        self.dec_conv2 = nn.Conv2d(2 * self.n_features, self.n_features, 3, padding=1)
        self.upsample3 = nn.Upsample((self.h, self.w))  # in_size // 2 -> in_size
        self.dec_conv3 = nn.Conv2d(2 * self.n_features, 1, 3, padding=1)

    def forward(self, x):
        # encoder
        e0 = F.relu(self.enc_conv0(x))
        e1 = F.relu(self.enc_conv1(self.pool0(e0)))
        e2 = F.relu(self.enc_conv2(self.pool1(e1)))
        e3 = F.relu(self.enc_conv3(self.pool2(e2)))

        # bottleneck
        b = F.relu(self.bottleneck_conv(self.pool3(e3)))

        # decoder
        d0 = F.relu(self.dec_conv0(torch.cat([self.upsample0(b), e3], 1)))
        d1 = F.relu(self.dec_conv1(torch.cat([self.upsample1(d0), e2], 1)))
        d2 = F.relu(self.dec_conv2(torch.cat([self.upsample2(d1), e1], 1)))
        d3 = self.dec_conv3(torch.cat([self.upsample3(d2), e0], 1))  # no activation
        return d3

class UNet(nn.Module):
    """UNet with convolutions (stride=2) for downsampling and transpose convolutinos (stride=2) for upsampling."""
    def __init__(self, in_channels=3, n_features=64):
        super().__init__()

        self.in_channels = in_channels
        self.n_features = n_features
        
        #TODO: We have issues with odd shape dimensions, we either need to handle this prior to model fitting or alter the code

        
        # encoder (downsampling)
        self.enc_block0 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.n_features, 3, padding=1),
            nn.BatchNorm2d(num_features=self.n_features),
            nn.ReLU(),
            nn.Conv2d(self.n_features, self.n_features, 3, padding=1),
            nn.BatchNorm2d(num_features=self.n_features),
            nn.ReLU()
        )
        self.pool0 = nn.Conv2d(self.n_features, self.n_features, 3, stride=2, padding=1)  
        self.enc_block1 = nn.Sequential(
            nn.Conv2d(self.n_features, 2 * self.n_features, 3, padding=1),
            nn.BatchNorm2d(num_features=2 * self.n_features),
            nn.ReLU(),
            nn.Conv2d(2 * self.n_features, 2 * self.n_features, 3, padding=1),
            nn.BatchNorm2d(num_features=2 * self.n_features),
            nn.ReLU()
        )
        self.pool1 = nn.Conv2d(2 * self.n_features, 2 * self.n_features, 3, stride=2, padding=1) 
        self.enc_block2 = nn.Sequential(
            nn.Conv2d(2 * self.n_features, 4 * self.n_features, 3, padding=1),
            nn.BatchNorm2d(num_features=4 * self.n_features),
            nn.ReLU(),
            nn.Conv2d(4 * self.n_features, 4 * self.n_features, 3, padding=1),
            nn.BatchNorm2d(num_features=4 * self.n_features),
            nn.ReLU()
        )
        self.pool2 = nn.Conv2d(4 * self.n_features, 4 * self.n_features, 3, stride=2, padding=1) 
        self.enc_block3 = nn.Sequential(
            nn.Conv2d(4 * self.n_features, 8 * self.n_features, 3, padding=1),
            nn.BatchNorm2d(num_features=8 * self.n_features),
            nn.ReLU(),
            nn.Conv2d(8 * self.n_features, 8 * self.n_features, 3, padding=1),
            nn.BatchNorm2d(num_features=8 * self.n_features),
            nn.ReLU()
        )
        self.pool3 = nn.Conv2d(8 * self.n_features, 8 * self.n_features, 3, stride=2, padding=1)

        # bottleneck
        self.bottleneck_block = nn.Sequential(
            nn.Conv2d(8 * self.n_features, 16 * self.n_features, 3, padding=1),
            nn.BatchNorm2d(num_features=16 * self.n_features),
            nn.ReLU(),
            nn.Conv2d(16 * self.n_features, 16 * self.n_features, 3, padding=1),
            nn.BatchNorm2d(num_features=16 * self.n_features),
            nn.ReLU()
        )

        # decoder (upsampling)
        self.upsample0 = nn.ConvTranspose2d(16 * self.n_features, 8 * self.n_features, 3, stride=2, padding=1, output_padding=1) 
        self.dec_block0 = nn.Sequential(
            nn.Conv2d(2 * (8 * self.n_features), 8 * self.n_features, 3, padding=1),
            nn.BatchNorm2d(num_features=8 * self.n_features),
            nn.ReLU(),
            nn.Conv2d(8 * self.n_features, 8 * self.n_features, 3, padding=1),
            nn.BatchNorm2d(num_features=8 * self.n_features),
            nn.ReLU()
        )
        self.upsample1 = nn.ConvTranspose2d(8 * self.n_features, 4 * self.n_features, 3, stride=2, padding=1, output_padding=1)
        self.dec_block1 = nn.Sequential(
            nn.Conv2d(2 * (4 * self.n_features), 4 * self.n_features, 3, padding=1),
            nn.BatchNorm2d(num_features=4 * self.n_features),
            nn.ReLU(),
            nn.Conv2d(4 * self.n_features, 4 * self.n_features, 3, padding=1),
            nn.BatchNorm2d(num_features=4 * self.n_features),
            nn.ReLU()
        )
        self.upsample2 = nn.ConvTranspose2d(4 * self.n_features, 2 * self.n_features, 3, stride=2, padding=1, output_padding=1)
        self.dec_block2 = nn.Sequential(
            nn.Conv2d(2 * (2 * self.n_features), 2 * self.n_features, 3, padding=1),
            nn.BatchNorm2d(num_features=2 * self.n_features),
            nn.ReLU(),
            nn.Conv2d(2 * self.n_features, 2 * self.n_features, 3, padding=1),
            nn.BatchNorm2d(num_features=2 * self.n_features),
            nn.ReLU()
        )
        self.upsample3 = nn.ConvTranspose2d(2 * self.n_features, 1 * self.n_features, 3, stride=2, padding=1, output_padding=1)
        self.dec_block3 = nn.Sequential(
            nn.Conv2d(2 * self.n_features, self.n_features, 3, padding=1),
            nn.BatchNorm2d(num_features=self.n_features),
            nn.ReLU(),
            nn.Conv2d(self.n_features, self.n_features, 3, padding=1),
            nn.BatchNorm2d(num_features=self.n_features),
            nn.ReLU(),
            nn.Conv2d(self.n_features, 1, 3, padding=1)
        )

    def forward(self, x):
        # encoder
        e0 = self.enc_block0(x)
        e1 = self.enc_block1(self.pool0(e0))
        e2 = self.enc_block2(self.pool1(e1))
        e3 = self.enc_block3(self.pool2(e2))

        # bottleneck
        b = self.bottleneck_block(self.pool3(e3))

        # decoder
        d0 = self.dec_block0(torch.cat([self.upsample0(b), e3], 1))
        d1 = self.dec_block1(torch.cat([self.upsample1(d0), e2], 1))
        d2 = self.dec_block2(torch.cat([self.upsample2(d1), e1], 1))
        d3 = self.dec_block3(torch.cat([self.upsample3(d2), e0], 1)) # no activation
        return d3
