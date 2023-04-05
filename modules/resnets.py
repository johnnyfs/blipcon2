from enum import Enum
from typing import Optional

import torch
from torch import nn


class DownSampleType(Enum):
    AVG_POOL= 0
    MAX_POOL = 1
    CONV = 2


class UpSampleType(Enum):
    NEAREST = 0
    BILINEAR = 1
    CONV = 2
    TRANSPOSE = 3


class NonLinearity(Enum):
    RELU = 0
    SIGMOID = 1
    TANH = 2
    LEAKY_RELU = 3
    ELU = 4
    SILU = 5


class MiniResNet(nn.Module):
    """
    Modeled after a (much simplified version) of stable diffusion's variation
    autoencoder's resnet blocks.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 norm_groups: int,
                 nonlinearity: NonLinearity=NonLinearity.RELU,
                 dropout: float=0.0,
                 residual: float=1.0):
        super().__init__()
        self.residual = residual

        if nonlinearity == NonLinearity.RELU:
            self.act = nn.ReLU()
        elif nonlinearity == NonLinearity.SILU:
            self.act = nn.SiLU()
        else:
            raise NotImplementedError(f"Non-linearity {nonlinearity} not implemented.")

        self.norm = nn.GroupNorm(norm_groups, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        else:
            self.shortcut = nn.Identity()

        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()

    def forward(self, x: torch.Tensor):
        input = x
        x = self.act(self.norm(x))
        x = self.act(self.conv(x))
        x = self.dropout(x)

        return x + self.residual * self.shortcut(input)


class MiniDownBlock(nn.Module):
    """
    Again cribbing from stable diffusion. Composes a resnet with a configurable
    downsample.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 norm_groups: int,
                 num_layers: int=1,
                 down_type: Optional[DownSampleType] = None,
                 nonlinearity: NonLinearity=NonLinearity.RELU,
                 dropout: float=0.0,
                 residual: float=1.0):
        super().__init__()
        self.layers = nn.ModuleList([
            MiniResNet(in_channels, out_channels, norm_groups)
        ])
        for i in range(num_layers - 1):
            self.layers.append(MiniResNet(out_channels, out_channels, norm_groups, nonlinearity, dropout, residual))

        if down_type == DownSampleType.CONV:
            self.down = nn.Conv2d(out_channels, out_channels, 3, 2, 1)
        elif down_type == DownSampleType.MAX_POOL:
            self.down = nn.MaxPool2d(2)
        elif down_type == DownSampleType.AVG_POOL:
            self.down = nn.AvgPool2d(2)
        elif down_type is None:
            self.down = nn.Identity()

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)

        return self.down(x)
    

class MiniUpBlock(nn.Module):
    """
    Good artists borrow.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 norm_groups: int,
                 num_layers: int=1,
                 up_type: Optional[UpSampleType]=None,
                 nonlinearity: NonLinearity=NonLinearity.RELU,
                 dropout: float=0.0,
                 residual: float=1.0):
        super().__init__()
        self.layers = nn.ModuleList([
            MiniResNet(in_channels, out_channels, norm_groups, residual=residual, dropout=dropout, nonlinearity=nonlinearity)
        ])
        for _ in range(num_layers - 1):
            self.layers.append(MiniResNet(out_channels, out_channels, norm_groups, residual=residual, dropout=dropout, nonlinearity=nonlinearity))

        if up_type == UpSampleType.NEAREST:
            self.up = nn.Upsample(scale_factor=2, mode='nearest')
        elif up_type == UpSampleType.BILINEAR:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        elif up_type == UpSampleType.CONV:
            self.up = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        elif up_type == UpSampleType.TRANSPOSE:
            self.up = nn.ConvTranspose2d(out_channels, out_channels, 3, 2, 1, 1)
        elif up_type is None:
            self.up = nn.Identity()

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)

        return self.up(x)