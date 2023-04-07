from typing import Optional

from torch import nn

from .enums import *
from .resnets import *


class MiniDownBlock(nn.Module):
    """
    Again cribbing from stable diffusion. Composes a resnet with a configurable
    downsample.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 norm_groups: int=1,
                 num_layers: int=1,
                 down_type: Optional[DownSample]=None,
                 nonlinearity: Optional[NonLinearity]=NonLinearity.RELU,
                 dropout: float=0.0,
                 residual: float=1.0):
        super().__init__()
        self.layers = nn.ModuleList([
            MiniResNet(in_channels,
                       out_channels,
                       norm_groups,
                       nonlinearity,
                       dropout,
                       residual)
        ])
        for i in range(num_layers - 1):
            self.layers.append(MiniResNet(out_channels,
                                          out_channels,
                                          norm_groups,
                                          nonlinearity,
                                          dropout,
                                          residual))

        self.down = get_module_for(down_type, channels=out_channels)

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
                 up_type: Optional[UpSample]=None,
                 nonlinearity: Optional[NonLinearity]=NonLinearity.RELU,
                 dropout: float=0.0,
                 residual: float=1.0):
        super().__init__()
        self.layers = nn.ModuleList([
            MiniResNet(in_channels,
                       out_channels,
                       norm_groups,
                       nonlinearity=nonlinearity,
                       residual=residual,
                       dropout=dropout)
        ])
        for _ in range(num_layers - 1):
            self.layers.append(MiniResNet(out_channels,
                                          out_channels,
                                          norm_groups,
                                          nonlinearity=nonlinearity,
                                          dropout=dropout,
                                          residual=residual))

        self.up = get_module_for(up_type, channels=out_channels)

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)

        return self.up(x)

  
class MiniMid(nn.Module):
    def __init__(self, 
                 channels: int,
                 norm_groups: int,
                 num_layers: int=1,
                 nonlinearity: Optional[NonLinearity]=NonLinearity.RELU,
                 dropout: float=0.0,
                 residual: float=1.0):
        super().__init__()
        self.layers = nn.ModuleList([
            MiniResNet(channels,
                       channels,
                       norm_groups,
                       nonlinearity=nonlinearity,
                       dropout=dropout,
                       residual=residual)
        ])
        for i in range(num_layers - 1):
            self.layers.append(MiniAttention(channels,
                                             norm_groups,
                                             nonlinearity=nonlinearity,
                                             residual=residual))
            self.layers.append(MiniResNet(channels,
                                          channels,
                                          norm_groups,
                                          nonlinearity=nonlinearity,
                                          dropout=dropout,
                                          residual=residual))

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)

        return x
