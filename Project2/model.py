import torch
import torch.nn as nn
import torch.nn.functional as F


class EncDec(nn.Module):

    def __init__(self, in_channels=3, n_features=64, in_size=128):
        super().__init__()

        self.in_channels = in_channels
        self.n_features = n_features
        self.in_size = in_size

        # encoder (downsampling)
        self.enc_conv0 = nn.Conv2d(self.in_channels, self.n_features, 3, padding=1)
        self.pool0 = nn.MaxPool2d(2, 2)  # in_size -> in_size // 2
        self.enc_conv1 = nn.Conv2d(self.n_features, self.n_features, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)  # in_size // 2 -> in_size // 4
        self.enc_conv2 = nn.Conv2d(self.n_features, self.n_features, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)  # in_size // 4 -> in_size // 8
        self.enc_conv3 = nn.Conv2d(self.n_features, self.n_features, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)  # in_size // 8 -> in_size // 16

        # bottleneck
        self.bottleneck_conv = nn.Conv2d(self.n_features, self.n_features, 3, padding=1)

        # decoder (upsampling)
        self.upsample0 = nn.Upsample(self.in_size // 8)  # in_size // 16 -> in_size // 8
        self.dec_conv0 = nn.Conv2d(self.n_features, self.n_features, 3, padding=1)
        self.upsample1 = nn.Upsample(self.in_size // 4)  # in_size // 8 -> in_size // 4
        self.dec_conv1 = nn.Conv2d(self.n_features, self.n_features, 3, padding=1)
        self.upsample2 = nn.Upsample(self.in_size // 2)  # in_size // 4 -> in_size // 2
        self.dec_conv2 = nn.Conv2d(self.n_features, self.n_features, 3, padding=1)
        self.upsample3 = nn.Upsample(self.in_size)  # in_size // 2 -> in_size
        self.dec_conv3 = nn.Conv2d(self.n_features, 1, 3, padding=1)

    def forward(self, x):
        # encoder
        e0 = self.pool0(F.relu(self.enc_conv0(x)))
        e1 = self.pool1(F.relu(self.enc_conv1(e0)))
        e2 = self.pool2(F.relu(self.enc_conv2(e1)))
        e3 = self.pool3(F.relu(self.enc_conv3(e2)))

        # bottleneck
        b = F.relu(self.bottleneck_conv(e3))

        # decoder
        d0 = F.relu(self.dec_conv0(self.upsample0(b)))
        d1 = F.relu(self.dec_conv1(self.upsample1(d0)))
        d2 = F.relu(self.dec_conv2(self.upsample2(d1)))
        d3 = self.dec_conv3(self.upsample3(d2))  # no activation
        return d3

class UNet_base(nn.Module):
    """Original UNet implementation with maxpooling and upsampling."""
    def __init__(self, n_features=64, in_size=128):
        super().__init__()

        self.n_features = n_features
        self.in_size = 128

        # encoder (downsampling)
        self.enc_conv0 = nn.Conv2d(3, self.n_features, 3, padding=1)
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
        self.upsample0 = nn.Upsample(self.in_size // 8)  # in_size // 16 -> in_size // 8
        self.dec_conv0 = nn.Conv2d(2 * self.n_features, self.n_features, 3, padding=1)
        self.upsample1 = nn.Upsample(self.in_size // 4)  # in_size // 8 -> in_size // 4
        self.dec_conv1 = nn.Conv2d(2 * self.n_features, self.n_features, 3, padding=1)
        self.upsample2 = nn.Upsample(self.in_size // 2)  # in_size // 4 -> in_size // 2
        self.dec_conv2 = nn.Conv2d(2 * self.n_features, self.n_features, 3, padding=1)
        self.upsample3 = nn.Upsample(self.in_size)  # in_size // 2 -> in_size
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
    def __init__(self, n_features=64):
        super().__init__()
        
        self.n_features = n_features

        # encoder (downsampling)
        self.enc_conv0 = nn.Conv2d(3, self.n_features, 3, padding=1)
        self.pool0 = nn.Conv2d(self.n_features, self.n_features, 3, stride=2, padding=1)  # 128 -> self.n_features
        self.enc_conv1 = nn.Conv2d(self.n_features, self.n_features, 3, padding=1)
        self.pool1 = nn.Conv2d(self.n_features, self.n_features, 3, stride=2, padding=1)  # self.n_features -> 32
        self.enc_conv2 = nn.Conv2d(self.n_features, self.n_features, 3, padding=1)
        self.pool2 = nn.Conv2d(self.n_features, self.n_features, 3, stride=2, padding=1)  # 32 -> 16
        self.enc_conv3 = nn.Conv2d(self.n_features, self.n_features, 3, padding=1)
        self.pool3 = nn.Conv2d(self.n_features, self.n_features, 3, stride=2, padding=1)  # 16 -> 8

        # bottleneck
        self.bottleneck_conv = nn.Conv2d(self.n_features, self.n_features, 3, padding=1)

        # decoder (upsampling)
        self.upsample0 = nn.ConvTranspose2d(self.n_features, self.n_features, 3, stride=2, padding=1, output_padding=1)  # 8 -> 16
        self.dec_conv0 = nn.Conv2d(2 * self.n_features, self.n_features, 3, padding=1)
        self.upsample1 = nn.ConvTranspose2d(self.n_features, self.n_features, 3, stride=2, padding=1, output_padding=1)  # 16 -> 32
        self.dec_conv1 = nn.Conv2d(2 * self.n_features, self.n_features, 3, padding=1)
        self.upsample2 = nn.ConvTranspose2d(self.n_features, self.n_features, 3, stride=2, padding=1, output_padding=1) # 32 -> self.n_features
        self.dec_conv2 = nn.Conv2d(2 * self.n_features, self.n_features, 3, padding=1)
        self.upsample3 = nn.ConvTranspose2d(self.n_features, self.n_features, 3, stride=2, padding=1, output_padding=1)  # self.n_features -> 128
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
