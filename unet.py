__version__ = '0.1'

import torch
import torch.nn as nn
import torchvision
from torchinfo import summary

def DoubleConv(in_channel, out_channel, kernel_size):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(num_features=out_channel),
        nn.ReLU(inplace=True),

        nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=kernel_size, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(num_features=out_channel),
        nn.ReLU(inplace=True)
    )

def UpSample(in_channel, out_channel, kernel_size, use_transpose=False):
    if use_transpose:
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_channel),
            nn.ReLU(inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(num_features=out_channel),
            nn.ReLU(inplace=True)
        )

class UNet(nn.Module):
    def __init__(self, in_channel, n_class):
        super(UNet, self).__init__()

        self.ch_sizes = [32, 64, 128, 256, 512]
        ### DOWN LAYERS
        # 1
        self.down1 = DoubleConv(in_channel=in_channel, out_channel=self.ch_sizes[0], kernel_size=3)
        self.downsample1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 2
        self.down2 = DoubleConv(in_channel=self.ch_sizes[0], out_channel=self.ch_sizes[1], kernel_size=3)
        self.downsample2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 3
        self.down3 = DoubleConv(in_channel=self.ch_sizes[1], out_channel=self.ch_sizes[2], kernel_size=3)
        self.downsample3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 4
        self.down4 = DoubleConv(in_channel=self.ch_sizes[2], out_channel=self.ch_sizes[3], kernel_size=3)
        self.downsample4 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Bottleneck
        self.bottleneck = DoubleConv(in_channel=self.ch_sizes[3], out_channel=self.ch_sizes[4], kernel_size=3)
        ### UPSAMPLE LAYERS 
        # IN REVERSE ORDER 4-3-2-1 TO MATCH DOWNSAMPLE LAYER IDS
        # 4
        self.up4 = UpSample(in_channel=self.ch_sizes[4], out_channel=self.ch_sizes[3], kernel_size=3)
        self.upconv4 = DoubleConv(in_channel=self.ch_sizes[4], out_channel=self.ch_sizes[3], kernel_size=3)
        # 3
        self.up3 = UpSample(in_channel=self.ch_sizes[3], out_channel=self.ch_sizes[2], kernel_size=3)
        self.upconv3 = DoubleConv(in_channel=self.ch_sizes[3], out_channel=self.ch_sizes[2], kernel_size=3)
        # 2 
        self.up2 = UpSample(in_channel=self.ch_sizes[2], out_channel=self.ch_sizes[1], kernel_size=3)
        self.upconv2 = DoubleConv(in_channel=self.ch_sizes[2], out_channel=self.ch_sizes[1], kernel_size=3)
        # 1
        self.up1 = UpSample(in_channel=self.ch_sizes[1], out_channel=self.ch_sizes[0], kernel_size=3)
        self.upconv1 = DoubleConv(in_channel=self.ch_sizes[1], out_channel=self.ch_sizes[0], kernel_size=3)

        # Output layer
        self.out = nn.Conv2d(in_channels=self.ch_sizes[0], out_channels=n_class, kernel_size=1)


    def forward(self, x):
        d1 = self.down1(x)
        x = self.downsample1(d1)

        d2 = self.down2(x)
        x = self.downsample2(d2)

        d3 = self.down3(x)
        x = self.downsample3(d3)

        d4 = self.down4(x)
        x = self.downsample4(d4)

        x = self.bottleneck(x)

        u4 = self.up4(x)
        x = self.upconv4(torch.concat([d4, u4], dim=1))

        u3 = self.up3(x)
        x = self.upconv3(torch.concat([d3, u3], dim=1))

        u2 = self.up2(x)
        x = self.upconv2(torch.concat([d2, u2], dim=1))

        u1 = self.up1(x)
        x = self.upconv1(torch.concat([d1, u1], dim=1))

        return self.out(x)


#model = UNet(in_channel=3, n_class=100)
#summary(model,input_size=((1, 3, 512, 512)))