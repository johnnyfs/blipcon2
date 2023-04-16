from typing import List

from torch import nn

from .blocks import *
from .dist import MiniDiag


class MiniEncoder(nn.Module):
    def __init__(self,
                 in_channels: int,
                 steps: List[int],
                 norm_groups: int,
                 layers_per_block: int=1,
                 nonlinearity: Optional[NonLinearity]=NonLinearity.SILU,
                 down_type: Optional[DownSample]=DownSample.CONV,
                 dropout: float=0.0,
                 residual: float=1.0):
        super().__init__()
        prev = steps[0]
        self.conv = nn.Conv2d(in_channels, prev, 3, 1, 1)
        print('adding step from {} to {}'.format(in_channels, prev))
        self.down = nn.ModuleList()

        for i, step in enumerate(steps):
            prev = steps[i - 1] if i > 0 else prev
            last = i == len(steps) - 1
            print('adding step from {} to {} with down_type {}'.format(prev, step, down_type if not last else 'None'))
            self.down.append(MiniDownBlock(prev,
                                           step,
                                           norm_groups,
                                           num_layers=layers_per_block,
                                           down_type=down_type if not last else None,
                                           nonlinearity=nonlinearity,
                                           dropout=dropout,
                                           residual=residual))
            prev = step

        self.mid = MiniMid(prev,
                           norm_groups,
                           num_layers=layers_per_block,
                           nonlinearity=nonlinearity,
                           dropout=dropout,
                           residual=residual)

    def forward(self, x: torch.Tensor):
        x = self.conv(x)

        for layer in self.down:
            x = layer(x)

        x = self.mid(x)

        return x


class MiniDecoder(nn.Module):
    def __init__(self,
                 in_channels: int,
                 steps: List[int],
                 norm_groups: int,
                 layers_per_block: int=1,
                 nonlinearity: Optional[NonLinearity]=NonLinearity.SILU,
                 up_type: Optional[UpSample]=UpSample.NEAREST,
                 dropout: float=0.0,
                 residual: float=1.0):
        super().__init__()

        prev = steps[0]
        self.conv = nn.Conv2d(in_channels, prev, 3, 1, 1)
        print('adding step from {} to {}'.format(in_channels, prev))
        self.mid = MiniMid(prev,
                           norm_groups,
                           num_layers=layers_per_block,
                           nonlinearity=nonlinearity,
                           dropout=dropout,
                           residual=residual)

        self.up = nn.ModuleList()

        for i, step in enumerate(steps):
            last = i == len(steps) - 1
            prev = steps[i - 1] if i > 0 else prev
            print('adding step from {} to {} with up_type {}'.format(prev, step, up_type if not last else 'None'))
            self.up.append(MiniUpBlock(prev,
                                       step,
                                       norm_groups,
                                       num_layers=layers_per_block,
                                       up_type=up_type if not last else None))
            prev = step

        self.norm = nn.GroupNorm(norm_groups, prev)
        self.act = get_module_for(nonlinearity)

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x = self.mid(x)
        for layer in self.up:
            x = layer(x)

        x = self.act(self.norm(x))

        return x


class MiniVae(nn.Module):
    class SampleMode(Enum):
        RANDOM = 0
        MEAN = 1

    def __init__(self,
                 in_channels: int,
                 steps: List[int],
                 norm_groups: int,
                 latent_channels: int,
                 layers_per_block: int=1,
                 nonlinearity: Optional[NonLinearity]=NonLinearity.SILU,
                 dropout: float=0.0,
                 residual: float=1.0,
                 down_type: Optional[DownSample]=DownSample.CONV,
                 up_type: Optional[UpSample]=UpSample.NEAREST,
                 generator: Optional[torch.Generator]=None):
        super().__init__()
        if len(steps) == 0:
            raise ValueError('steps must be non-empty')
            
        self.encoder = MiniEncoder(in_channels,
                                   steps,
                                   norm_groups,
                                   layers_per_block=layers_per_block,
                                   nonlinearity=nonlinearity,
                                   down_type=down_type,
                                   dropout=dropout,
                                   residual=residual)
        self.norm = nn.GroupNorm(norm_groups, steps[-1])
        self.act = get_module_for(nonlinearity)
        self.quant_conv = nn.Conv2d(steps[-1], latent_channels * 2, 3, 1, 1)

        self.decoder = MiniDecoder(latent_channels,
                                   list(reversed(steps)),
                                   norm_groups,
                                   layers_per_block=layers_per_block,
                                   nonlinearity=nonlinearity,
                                   up_type=up_type,
                                   dropout=dropout,
                                   residual=residual)
        self.conv_out = nn.Conv2d(steps[0], in_channels, 3, 1, 1)
        self.generator = generator
        self.sample_mode = MiniVae.SampleMode.RANDOM
        self.sigmoid = nn.Sigmoid()

    def set_mean_mode(self):
        self.sample_mode = MiniVae.SampleMode.MEAN

    def set_sample_mode(self):
        self.sample_mode = MiniVae.SampleMode.RANDOM

    def toggle_sample_mode(self):
        if self.sample_mode == MiniVae.SampleMode.RANDOM:
            self.sample_mode = MiniVae.SampleMode.MEAN
        else:
            self.sample_mode = MiniVae.SampleMode.RANDOM

    def encode(self, x: torch.Tensor):
        x = self.encoder(x)
        x = self.act(self.norm(x))
        x = self.quant_conv(x)
        posterior = MiniDiag(x)

        return posterior

    def decode(self, x: torch.Tensor):
        x = self.decoder(x)
        x = self.conv_out(x)
        x = self.sigmoid(x)

        return x

    def forward(self, x: torch.Tensor):
        posterior = self.encode(x)
        if self.sample_mode == MiniVae.SampleMode.RANDOM:
            sample = posterior.sample(self.generator)
        else:
            sample = posterior.mean
        reconstructed = self.decode(sample)

        return reconstructed
    
class ResNetAE(nn.Module):
    """
    A simple autoencoder w/o a variational step that has demonstrated
    decent performance on 8-bit game images.
    """
    def __init__(self,
                 input_channels=3,
                 layers=[64, 128, 256, 256],
                 norm_groups=8,
                 nonlinearity=NonLinearity.RELU,
                 dropout=0.0,
                 residual=1.0,
                 latent_channels=8,
                 layers_per_block=1,
                 temperature=2.0):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, layers[0], 3, padding=1),
        )
        for i in range(len(layers)):
            prev = layers[i-1] if i > 0 else layers[0]
            is_last = i == len(layers) - 1
            down = DownSample.CONV if not is_last else None
            print('adding step from {} to {} with down={}'.format(prev, layers[i], down))
            block = MiniDownBlock(prev,
                                  layers[i],
                                  norm_groups=norm_groups,
                                  num_layers=layers_per_block,
                                  nonlinearity=nonlinearity,
                                  down_type=down,
                                  dropout=dropout,
                                  residual=residual)
            self.encoder.add_module(f'dblock{i}', block)
        self.mid = nn.Sequential(
            MiniMid(layers[-1],
                            norm_groups=norm_groups,
                            num_layers=2,
                            nonlinearity=nonlinearity,
                            dropout=dropout,
                            residual=residual),
            nn.GroupNorm(norm_groups, layers[-1]),
            get_module_for(nonlinearity),
            nn.Conv2d(layers[-1], latent_channels, 1),
            nn.Conv2d(latent_channels, layers[-1], 1),
            MiniMid(layers[-1],
                           norm_groups=norm_groups,
                           num_layers=2,
                           nonlinearity=nonlinearity,
                           dropout=dropout,
                           residual=residual)
        )

        self.decoder = nn.Sequential()
        for i in range(len(layers) - 1, -1, -1):
            next = layers[i-1] if i > 0 else layers[0]
            is_last = i == 0
            up = UpSample.TRANSPOSE if not is_last else None
            print('adding step from {} to {} with up={}'.format(layers[i], next, up))
            block = MiniUpBlock(layers[i],
                                next,
                                norm_groups=norm_groups,
                                num_layers=layers_per_block,
                                nonlinearity=nonlinearity,
                                up_type=up,
                                dropout=dropout,
                                residual=residual)
            self.decoder.add_module(f'ublock{i}', block)

        self.decoder.add_module('final', nn.Conv2d(layers[0], input_channels, 3, padding=1))
        self.decoder.add_module('sigmoid', nn.Sigmoid())
        
    def set_mean_mode(self):
        pass
    
    def set_sample_mode(self):
        pass

    def forward(self, x):
        names, patterns, attributes, palettes = self.encoder(x)
        return self.decoder(names, patterns, attributes, palettes)

    def set_temperature(self, temperature, hard=False):
        self.decoder.set_temperature(temperature, hard)