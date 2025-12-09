# -- coding: utf-8 --
import torch.nn as nn
from torch.nn import functional as F
import torch
import torch.optim as optim
import math
from copy import deepcopy

class UNet(nn.Module):
    """
    UNet backbone with encoder-decoder architecture
    """
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = in_channels
        self.n_classes = out_channels
        self.bilinear = bilinear
        self.feas = []
 
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256) 
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
    
        self.outc = OutConv(64, out_channels)

    def forward(self, x):
        self.feas.clear()
 
        x1 = self.inc(x)
        self.feas.append(x1)
        x2 = self.down1(x1)
        self.feas.append(x2)
        x3 = self.down2(x2)
        self.feas.append(x3)
        x4 = self.down3(x3)
        self.feas.append(x4)
        x5 = self.down4(x4)
        self.feas.append(x5)
        
        x = self.up1(x5, x4)
        self.feas.append(x)
        x = self.up2(x, x3)
        self.feas.append(x)
        x = self.up3(x, x2)
        self.feas.append(x)
        x = self.up4(x, x1)
        self.feas.append(x)
        
        x = self.outc(x)
        self.feas.append(x)

        return x

class DoubleConv(nn.Module):
    """Double convolution block with batch normalization"""
    def __init__(self, in_channels, out_channels, use_bn=True):
        super().__init__()
        self.use_bn = use_bn

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        if self.use_bn: 
            self.bn1 = nn.BatchNorm2d(out_channels)
        else:
            self.bn1 = nn.BatchNorm2d(out_channels, track_running_stats=False, affine=False)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        if self.use_bn:
            self.bn2 = nn.BatchNorm2d(out_channels)
        else:
            self.bn2 = nn.BatchNorm2d(out_channels, track_running_stats=False, affine=False)
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class Down(nn.Module):
    """Downsampling block with max pooling and double convolution"""
    def __init__(self, in_channels, out_channels, use_bn=True):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.double_conv = DoubleConv(in_channels, out_channels, use_bn=use_bn)

    def forward(self, x):
        x = self.maxpool(x)
        x = self.double_conv(x)
        return x

class Up(nn.Module):
    """Upsampling block with bilinear interpolation and double convolution"""
    def __init__(self, in_channels, out_channels, use_bn=True):
        super().__init__()
        self.use_bn = use_bn
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upconv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False)
        
        self.conv1 = nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1, bias=False)
        if self.use_bn:
            self.bn1 = nn.BatchNorm2d(out_channels)
        else:
            self.bn1 = nn.BatchNorm2d(out_channels, track_running_stats=False, affine=False)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        if self.use_bn:
            self.bn2 = nn.BatchNorm2d(out_channels)
        else:
            self.bn2 = nn.BatchNorm2d(out_channels, track_running_stats=False, affine=False)
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, decoder, skip):
        decoder = self.upconv(self.upsample(decoder))
        x = torch.cat([skip, decoder], dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class OutConv(nn.Module):
    """Output convolution layer"""
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        return self.conv(x)

def create_unet_model(in_channels=3, out_channels=1):
    """Create UNet model for segmentation tasks"""
    return UNet(in_channels=in_channels, out_channels=out_channels)