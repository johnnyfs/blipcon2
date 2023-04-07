from typing import Optional

import torch
from torch import nn

from .enums import *
from .resnets import *
from .attention import *


class MiniMid(nn.Module):
    def __init__(self, channels: int, norm_groups: int, num_layers: int=1):
        super().__init__()
        self.layers = nn.ModuleList([
            MiniResNet(channels, channels, norm_groups)
        ])
        for i in range(num_layers - 1):
            self.layers.append(MiniAttention(channels, norm_groups))
            self.layers.append(MiniResNet(channels, channels, norm_groups))

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)

        return x


class MiniResNet(nn.Module):
    """
    Modeled after a (much simplified version) of stable diffusion's variation
    autoencoder's resnet blocks.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 norm_groups: int,
                 nonlinearity: Optional[NonLinearity]=NonLinearity.RELU,
                 dropout: float=0.0,
                 residual: float=1.0):
        super().__init__()
        self.residual = residual
        self.act = get_module_for(nonlinearity)

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