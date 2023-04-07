from enum import Enum
from typing import Optional

from torch import nn


class ModuleEnum(Enum):
    def get(self, *args, **kwargs):
        raise NotImplementedError("Must implement ModuleEnum.get() method.")


class DownSample(ModuleEnum):
    AVG_POOL= 0
    MAX_POOL = 1
    CONV = 2

    def get(self, channels=None):
        if self == DownSample.AVG_POOL:
            return nn.AvgPool2d(2)
        elif self == DownSample.MAX_POOL:
            return nn.MaxPool2d(2)
        elif self == DownSample.CONV:
            return nn.Conv2d(channels, channels, 3, 2, 1)


class UpSample(ModuleEnum):
    NEAREST = 0
    BILINEAR = 1
    CONV = 2
    TRANSPOSE = 3
    
    def get(self, *, channels=None):
        if self == UpSample.NEAREST:
            return nn.Upsample(scale_factor=2, mode='nearest')
        elif self == UpSample.BILINEAR:
            return nn.Upsample(scale_factor=2, mode='bilinear')
        elif self == UpSample.CONV:
            return nn.Conv2d(channels, channels, 3, 1, 1)
        elif self == UpSample.TRANSPOSE:
            return nn.ConvTranspose2d(channels, channels, 3, 2, 1, 1)
 
 
class NonLinearity(ModuleEnum):
    RELU = 0
    SIGMOID = 1
    TANH = 2
    LEAKY_RELU = 3
    ELU = 4
    SILU = 5
    
    def get(self, negative_slope: float=None):
        if self == NonLinearity.RELU:
            return nn.ReLU()
        elif self == NonLinearity.SIGMOID:
            return nn.Sigmoid()
        elif self == NonLinearity.TANH:
            return nn.Tanh()
        elif self == NonLinearity.LEAKY_RELU:
            return nn.LeakyReLU(negative_slope=2.0)
        elif self == NonLinearity.ELU:
            return nn.ELU()
        elif self == NonLinearity.SILU:
            return nn.SiLU()


def get_module_for(enum_: Optional[ModuleEnum], *args, **kwargs):
    if enum_ is None:
        return nn.Identity()
    else:
        return enum_.get(*args, **kwargs)