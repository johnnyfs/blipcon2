from typing import Optional

import torch
from torch import nn

from modules.blocks import MiniDownBlock, MiniUpBlock
from modules.enums import DownSample, NonLinearity, UpSample, get_module_for
from modules.resnets import MiniResNet, MiniMid


class NESEncoder(nn.Module):
    def __init__(self,
                 nonlinearity: Optional[NonLinearity]=NonLinearity.RELU,
                 spatial_channels: int=64,
                 color_channels: int=8,
                 norm_groups: int=8):
        """
        The goal of the model is to learn
        - the set of 256 8x8 4-color monochromatic tiles from which the image
          is composed, called `patterns`
        - the distributions of those patterns across a 256x240 grid (which might
          be offset by up to 15 pixels), forming a table called `names`
        - the colors used to compose the image; there will be at most 32, organized
          into 8 4-color `palettes`
        - the distribution of the palettes across the image, called `attributes`
          each index 0-3 in the referenced `pattern` samples from the corresponding
          index given the value in a 16x15 grid of 16x16 pixel-sized celsl
          overlaying the image. Eg, the tile at (31,29) is in palette grid pos
          (15,14), so if that value is 3, hen for each index 0 in that position it
          samples from the first color in the third palette.
        """
        super().__init__()
        
        # common: (b, 3, 240, 256) -> (b, ch, 120, 128)
        self.spatial_common = nn.Sequential(
            nn.Conv2d(3, spatial_channels, 3, padding=1),
            MiniDownBlock(in_channels=spatial_channels,
                          out_channels=spatial_channels,
                          norm_groups=norm_groups,
                          num_layers=1,
                          nonlinearity=nonlinearity,
                          down_type=DownSample.AVG_POOL)
        )
        
        # patterns: (b, ch, 120, 128) -> (b, 4, 16*8, 16*8)
        pattern_channels = spatial_channels
        self.patterns = nn.Sequential(
            nn.UpsamplingBilinear2d((16*8, 16*8)),
            nn.Conv2d(spatial_channels, pattern_channels, 3, padding=1),
            nn.GroupNorm(4, pattern_channels),
            get_module_for(nonlinearity),
            nn.Conv2d(pattern_channels, 4, 3, padding=1),
            nn.GroupNorm(1, 4),
            get_module_for(nonlinearity),
        )
        
        # names: (b, ch, 240, 256) -> (b, ch, 30, 32)
        self.names = nn.Sequential(
            nn.Conv2d(spatial_channels, 128, 3, padding=1),
            nn.GroupNorm(4, 128),
            get_module_for(nonlinearity),
            nn.AvgPool2d(2, 2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.GroupNorm(4, 256),
            get_module_for(nonlinearity),
            nn.AvgPool2d(2, 2),
        )
        self.names_q = nn.Linear(30*32*256, 256)
        self.names_k = nn.Linear(30*32*256, 256)
        self.patterns_v = nn.Linear(4*16*8*16*8, 256)
        self.names_attention = nn.MultiheadAttention(256, 4)
        self.names_forward = nn.Linear(256, 30*32*256)
        
        # common: (b, 3, 240, 256) -> (b, ch, 30, 32)
        self.color_common = nn.Sequential(
            nn.Conv2d(3, color_channels, 3, padding=1),
            nn.GroupNorm(1, color_channels),
            get_module_for(nonlinearity),
            nn.AvgPool2d(8, 8)
        )
        
        # palettes: (b, ch, 30, 32) -> (b, ch, 4, 3)
        self.palettes = nn.Sequential(
            nn.Conv2d(color_channels, color_channels, 3),
            nn.GroupNorm(1, color_channels),
            get_module_for(nonlinearity),
            nn.AvgPool2d((7, 10))
        )
        
        # attributes: (b, ch, 30, 32) -> (b, ch, 15, 16)
        self.attributes = nn.Sequential(
            nn.Conv2d(color_channels, color_channels, 3, stride=2, padding=1),
            nn.GroupNorm(1, color_channels),
            get_module_for(nonlinearity)
        )
        
    def forward(self, x):
        spatial_common = self.spatial_common(x)
        patterns = self.patterns(spatial_common)
        
        names = self.names(spatial_common)
        q = self.names_q(names.flatten(1))
        k = self.names_k(names.flatten(1))
        v = self.patterns_v(patterns.flatten(1))
        names = self.names_attention(q, k, v)[0]
        names = self.names_forward(names).view(-1, 256, 30, 32)
        
        color_common = self.color_common(x)
        palettes = self.palettes(color_common)
        attributes = self.attributes(color_common)
        
        return names, patterns, attributes, palettes
            

class NESDecoder(nn.Module):
    def __init__(self, channels, nonlinearity=NonLinearity.SILU, dropout=0.0, residual=1.0, temperature=2.0, hard=False):
        super().__init__()
        # incoming shape is (b, ch, 30, 32)
        # No pre-processing or normalization at the first stage
        # It *seems* to degrade performance, but I don't know why.
        
        # Theory: applying self-attention to the incoming
        # data might help it learn what is common to the
        # spatial-v-visual tasks and the pixel-v-color tasks.
        self.tables_common = MiniMid(
                    channels=channels,
                    norm_groups=4,
                    nonlinearity=nonlinearity,
                    dropout=dropout,
                    residual=residual
                )
        self.cat_common = MiniMid(
                channels=channels,
                norm_groups=4,
                nonlinearity=nonlinearity,
                dropout=dropout,
                residual=residual,
                num_layers=2
            )
        self.pixels_common = MiniMid(
                channels=channels,
                norm_groups=4,
                nonlinearity=nonlinearity,
                dropout=dropout,
                residual=residual,
                num_layers=2
            )
        self.colors_common = MiniMid(
                channels=channels,
                norm_groups=4,
                nonlinearity=nonlinearity,
                dropout=dropout,
                residual=residual,
                num_layers=2
            )
        
        # names are 256 30x32, so only convolve _ -> 256
        self.to_names = nn.Sequential(
            MiniResNet(
                in_channels=channels,
                out_channels=256,
                norm_groups=4,
                nonlinearity=nonlinearity,
                dropout=dropout,
                residual=residual
            ),
            nn.GroupNorm(1, 256),
            get_module_for(nonlinearity)
        )
        
        # patterns are 256 8x8x4, so we need to downsample by 2x2
        # and convolve _ -> 256
        self.to_patterns = nn.Sequential(
            MiniUpBlock(
                in_channels=channels,
                out_channels=128,
                norm_groups=4,
                nonlinearity=nonlinearity,
                dropout=dropout,
                residual=residual,
                up_type=UpSample.TRANSPOSE
            ),
            MiniDownBlock(
                in_channels=128,
                out_channels=256,
                norm_groups=4,
                nonlinearity=nonlinearity,
                dropout=dropout,
                residual=residual,
                down_type=DownSample.CONV
            ),
            nn.Conv2d(256, 256, 3, stride=2, padding=(2, 1)),
            get_module_for(nonlinearity),
            nn.GroupNorm(1, 256),
            get_module_for(nonlinearity)
        )
        
        # Theory: as with the tables, applying self-attention
        # might help the model learn what is common to the
        # categorical tasks (deciding tiles and palettes)

        
        # attributes are 4 15x16 so we need to
        # downsample, convolve _ -> 4,
        # then redouble for compatibili
        self.to_attributes = nn.Sequential(
            MiniResNet(
                in_channels=channels,
                out_channels=128,
                norm_groups=4,
                nonlinearity=nonlinearity,
                dropout=dropout,
                residual=residual
            ),
            nn.AvgPool2d(2),
            nn.Conv2d(128, 4, 1),
            nn.GroupNorm(1, 4),
            get_module_for(nonlinearity)
        )
        
        # palettes are 4 4x3 so we need to downsample by 2x3
        self.to_palettes = nn.Sequential(
            MiniResNet(
                in_channels=channels,
                out_channels=128,
                norm_groups=4,
                nonlinearity=nonlinearity,
                dropout=dropout,
                residual=residual
            ),
            nn.Conv2d(128, 4, 2, stride=(2, 3), padding=(1, 2)),
            nn.Conv2d(4, 1, (2, 1), stride=(2, 1)),
            nn.AvgPool2d((2, 1)),
            nn.GroupNorm(1,1),
            get_module_for(nonlinearity)
        )
        
        # Image post-processing seems to do best
        # with just the sigmoid, but I worry
        # that it might not be desirable.
        self.temperature = temperature
        self.hard = hard
        self.final = nn.Sequential(
             nn.Sigmoid()
        )
        
    def set_temperature(self, temperature, hard=False):
        self.temperature = temperature
        self.hard = hard
   
    def forward(self, names, patterns, attributes, palettes):
        b = names.shape[0]
        
        # tables_common = self.tables_common(x)
        # cat_common = self.cat_common(x)
        # pixels_common = self.pixels_common(x)
        # colors_common = self.colors_common(x)
        
        # names = self.to_names(tables_common + pixels_common)
        names = names.view(b, 256, 30*32)
        names = names.permute(0, 2, 1)
        names_idxs = torch.nn.functional.gumbel_softmax(names, tau=self.temperature, hard=self.hard)
        
        
        # patterns = self.to_patterns(cat_common + pixels_common)
        # patterns = patterns.view(b, 256, 8*8, 4)
        
        patterns = patterns.view(b, 4, 16, 8, 16, 8)
        patterns = patterns.permute(0, 2, 4, 3, 5, 1).contiguous()
        patterns = patterns.softmax(dim=-1)
        #patterns = torch.nn.functional.gumbel_softmax(patterns, tau=self.temperature, hard=self.hard)
        patterns = patterns.view(b, 256, 8*8*4)
        
        # attributes = self.to_attributes(tables_common + colors_common)
        attributes = attributes.view(b, 8, 15*16)
        attributes = attributes.permute(0, 2, 1)
        # Use a lower temperature for the attributes, because
        # they have a lower cardinality
        attributes_idxs = torch.nn.functional.gumbel_softmax(attributes, tau=self.temperature, hard=self.hard)
        
        # palettes = self.to_palettes(cat_common + colors_common)
        palettes = palettes.view(b, 8, 4*3)
        
        
        # names: (b, 30*32, 256)
        # patterns: (b, 256, 8*8*4)
        # attributes: (b, 15*16, 8)
        # palettes: (b, 8, 4*3)
        
        monochrome = torch.matmul(names_idxs, patterns)
        colorization = torch.matmul(attributes_idxs, palettes)
        
        # monochrome: (b, 30*32, 8*8*4)
        # colorization: (b, 15*16, 4*3)
        
        monochrome = monochrome.view(b, 30*32, 8*8, 4, 1)
        colorization = colorization.view(b, 15, 16, 4, 3)
        colorization = colorization.repeat_interleave(repeats=2, dim=1).repeat_interleave(repeats=2, dim=2)
        colorization = colorization.view(b, 30*32, 1, 4, 3)

        colorized = (monochrome * colorization).sum(dim=3)
        
        # colorized: (b, 30*32, 8*8, 3)
        # convert to (b, 3, 30*8, 32*8)
        image = colorized.view(b, 30, 32, 8, 8, 3)
        image = image.permute(0, 5, 1, 3, 2, 4)
        image = image.reshape(b, 3, 30*8, 32*8)
        image = self.final(image)
        
        return image, names_idxs, patterns, attributes_idxs, palettes


class NESAE(nn.Module):
    """
    An attempt to make a model that learns to decode state
    into its underlying tile patterns, name tables, palettes,
    and colorization attributes. 
    
    It does not work.
    """
    def __init__(self,
                nonlinearity=NonLinearity.RELU,
                spatial_channels=64,
                color_channels=8,
                norm_groups=8,
                dropout=0.0,
                residual=1.0,
                temperature=1.0,
                hard=False):
        super().__init__()
        self.encoder = NESEncoder(
            nonlinearity=nonlinearity,
            spatial_channels=spatial_channels,
            color_channels=color_channels,
            norm_groups=norm_groups
        )
        self.decoder = NESDecoder(
            nonlinearity=nonlinearity,
            dropout=dropout,
            residual=residual,
            temperature=temperature,   
            hard=hard)
        
    def forward(self, x):
        names, patterns, attributes, palettes = self.encoder(x)
        return self.decoder(names, patterns, attributes, palettes)


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
                 layers_per_block=1):
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
    
    def encode(self, x):
        return self.encode(x)

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)

    def set_temperature(self, temperature, hard=False):
        self.decoder.set_temperature(temperature, hard)