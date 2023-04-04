import torch
from torch import nn


class MiniResNet(nn.Module):
    """
    Modeled after a (much simplified version) of stable diffusion's variation
    autoencoder's resnet blocks.
    """
    def __init__(self, in_channels: int, out_channels: int, norm_groups: int, dropout: float=0.0):
        super().__init__()
        self.act = nn.SiLU()

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

        return x + self.shortcut(input)


class MiniDownBlock(nn.Module):
    """
    Again cribbing from stable diffusion. Composes a a resnet block with a
    downsampling convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, norm_groups: int, num_layers: int=1, down: bool=True, dropout: float=0.0, use_conv=False):
        super().__init__()
        self.layers = nn.ModuleList([
            MiniResNet(in_channels, out_channels, norm_groups)
        ])
        for i in range(num_layers - 1):
            self.layers.append(MiniResNet(out_channels, out_channels, norm_groups, dropout))

        if down:
            if use_conv:
                self.down = nn.Conv2d(out_channels, out_channels, 3, 2, 1)
            else:
                self.down = nn.MaxPool2d(2)
        else:
            self.down = nn.Identity()

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)

        return self.down(x)
    

class MiniUpBlock(nn.Module):
    """
    Good artists borrow.
    """
    def __init__(self, in_channels: int, out_channels: int, norm_groups: int, num_layers: int=1, up: bool=True, use_transpose: bool=False, dropout: float=0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            MiniResNet(in_channels, out_channels, norm_groups)
        ])
        for i in range(num_layers - 1):
            self.layers.append(MiniResNet(out_channels, out_channels, norm_groups, dropout))

        if up:
            if not use_transpose:
                self.up = nn.Upsample(scale_factor=2, mode='nearest')
            else:
                self.up = nn.ConvTranspose2d(out_channels, out_channels, 3, 2, 1, 1)
        else:
            self.up = nn.Identity()

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)

        return self.up(x)