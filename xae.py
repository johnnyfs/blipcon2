"""
An exploratory ae project to attempt to
"map" the distribution of the "game states."

The goal is to determine the best structure
for a state predictor capable of reproducing
the game states perfectly.
"""

import os, pickle, random

from concurrent.futures import ThreadPoolExecutor

import torch
from torch import nn
from torchvision import transforms as T

import PIL
import pygame

from processing.states import sample_pickled_states
from modules.management import *
from modules.ae import *
from modules.blocks import MiniMid, MiniDownBlock, MiniUpBlock
from modules.enums import *
from visualization.images import pil_to_surface
from visualization.graphs import mk_pil_table


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


class ResNetAE(nn.Module):
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
        # self.encoder = nn.Sequential(
        #     nn.Conv2d(input_channels, layers[0], 3, padding=1),
        # )
        # for i in range(len(layers)):
        #     prev = layers[i-1] if i > 0 else layers[0]
        #     is_last = i == len(layers) - 1
        #     down = DownSample.CONV if not is_last else None
        #     print('adding step from {} to {} with down={}'.format(prev, layers[i], down))
        #     block = MiniDownBlock(prev,
        #                           layers[i],
        #                           norm_groups=norm_groups,
        #                           num_layers=layers_per_block,
        #                           nonlinearity=nonlinearity,
        #                           down_type=down,
        #                           dropout=dropout,
        #                           residual=residual)
        #     self.encoder.add_module(f'dblock{i}', block)
        # self.mid = nn.Sequential(
        #     MiniMid(layers[-1],
        #                     norm_groups=norm_groups,
        #                     num_layers=2,
        #                     nonlinearity=nonlinearity,
        #                     dropout=dropout,
        #                     residual=residual),
        #     nn.GroupNorm(norm_groups, layers[-1]),
        #     get_module_for(nonlinearity),
        #     nn.Conv2d(layers[-1], latent_channels, 1),
        #     nn.Conv2d(latent_channels, layers[-1], 1),
        #     MiniMid(layers[-1],
        #                    norm_groups=norm_groups,
        #                    num_layers=2,
        #                    nonlinearity=nonlinearity,
        #                    dropout=dropout,
        #                    residual=residual)
        # )
        # self.decoder = NESDecoder(channels=64)
        self.encoder = NESEncoder(nonlinearity=nonlinearity,
                                  spatial_channels=64,
                                  color_channels=8)
        self.decoder = NESDecoder(channels=layers[-1],
                                  nonlinearity=nonlinearity,
                                  dropout=dropout,
                                  residual=residual,
                                  temperature=temperature)
        # self.decoder = nn.Sequential()
        # for i in range(len(layers) - 1, -1, -1):
        #     next = layers[i-1] if i > 0 else layers[0]
        #     is_last = i == 0
        #     up = UpSample.TRANSPOSE if not is_last else None
        #     print('adding step from {} to {} with up={}'.format(layers[i], next, up))
        #     block = MiniUpBlock(layers[i],
        #                         next,
        #                         norm_groups=norm_groups,
        #                         num_layers=layers_per_block,
        #                         nonlinearity=nonlinearity,
        #                         up_type=up,
        #                         dropout=dropout,
        #                         residual=residual)
        #     self.decoder.add_module(f'ublock{i}', block)

        # self.decoder.add_module('final', nn.Conv2d(layers[0], input_channels, 3, padding=1))
        # self.decoder.add_module('sigmoid', nn.Sigmoid())
        
    def set_mean_mode(self):
        pass
    
    def set_sample_mode(self):
        pass

    def forward(self, x):
        names, patterns, attributes, palettes = self.encoder(x)
        return self.decoder(names, patterns, attributes, palettes)

    def set_temperature(self, temperature, hard=False):
        self.decoder.set_temperature(temperature, hard)


def make_frame(stage, screen, actual, predicted,
               names_idxs, patterns, attributes_idxs, palettes):
    img1 = actual.numpy() * 255
    img1 = img1.astype('uint8')
    img1 = pygame.surfarray.make_surface(img1)
    stage.blit(img1, (0, 0))
    img2 = predicted.numpy() * 255
    img2 = img2.astype('uint8')
    img2 = pygame.surfarray.make_surface(img2)
    stage.blit(img2, (256, 0))
    
    # Convert the 256x30x32 names to 30x32 with the index of the
    # max value from the 256
    names_idxs = names_idxs.view(256, 30, 32).permute(2, 1, 0).argmax(dim=2)
    names_idxs = names_idxs.unsqueeze(-1).repeat_interleave(3, dim=-1)
    img3 = names_idxs.numpy().astype('uint8')
    # Convert numpy to PIL monocrhome image
    img3 = PIL.Image.fromarray(img3, mode='RGB')
    img3 = pil_to_surface(img3)
    img3 = pygame.transform.scale(img3, (32*8, 30*8))
    stage.blit(img3, (0, 240))
    
    # patterns are 256x8x84
    patterns = patterns.view(16, 16, 8, 8, 4).argmax(dim=4)
    patterns = patterns.unsqueeze(-1).repeat_interleave(3, dim=-1)
    patterns = patterns.permute(0, 2, 1, 3, 4).contiguous()
    patterns = patterns.view(16*8, 16*8, 3)
    img4 = patterns.numpy().astype('uint8') * 85
    img4 = PIL.Image.fromarray(img4, mode='RGB')
    img4 = pil_to_surface(img4)
    img4 = pygame.transform.scale(img4, (16*8*2, 16*8*2))
    stage.blit(img4, (256, 240))
    
    # attributes are 4x15x16
    attributes_idxs = attributes_idxs.view(8, 15, 16).permute(2, 1, 0).argmax(dim=2)
    attributes_idxs = attributes_idxs.unsqueeze(-1).repeat_interleave(3, dim=-1)
    img5 = attributes_idxs.numpy().astype('uint8') * 36
    img5 = PIL.Image.fromarray(img5, mode='RGB')
    img5 = pil_to_surface(img5)
    img5 = pygame.transform.scale(img5, (32*8, 30*8))
    stage.blit(img5, (0, 240 + 256))
    
    # palettes are 4,4*3
    palettes = palettes.view(8, 4, 3)
    img6 = (palettes * 255).numpy().astype('uint8')
    img6 = PIL.Image.fromarray(img6, mode='RGB')
    img6 = pil_to_surface(img6)
    img6 = pygame.transform.scale(img6, (8*16, 4*16))
    stage.blit(img6, (256, 240 + 256))
    
    pygame.transform.scale(stage, (stage.get_width() * SCALE, stage.get_height() * SCALE), screen)
    pygame.display.flip()
                              

BATCH_SIZE=16
DROPOUT=0.5
LEARNING_RATE=1e-5
WEIGHT_DECAY=1e-5
TEMPERATURE=2.0
ACT=NonLinearity.SIGMOID

DISPLAY=True
SCALE=1
RELOAD=False
VARIATIONAL=False

if not VARIATIONAL:
    model: nn.Module = ResNetAE(
        input_channels=3,
        layers=[64, 128, 256, 256],
        dropout=DROPOUT,
        nonlinearity=ACT,
        layers_per_block=1,
        latent_channels=8,
        residual=1.3333,
        temperature=TEMPERATURE).to("cuda")
else:
    model: nn.Module = MiniVae(
        3, [32, 64, 128, 128],
        norm_groups=4,
        latent_channels=8,
        layers_per_block=1,
        dropout=DROPOUT,
        up_type=UpSample.TRANSPOSE,
        residual=1.333).to("cuda")
    model.set_mean_mode()

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
loss_fn = nn.MSELoss()

if DISPLAY:
    state_w, state_h = 256, 240
    stage_w, stage_h = 256 * 2, 240 * 2 + 256
    screen = pygame.display.set_mode((stage_w * SCALE, stage_h * SCALE))
    stage = pygame.Surface((stage_w, stage_h))

if RELOAD:
    try:
        print('Loading weights')
        state = torch.load('data/training/base_ae.pth')
        epoch = state['epoch']
        training_losses = state['training_losses']
        eval_losses = state['eval_losses']
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
    except:
        print('No weights found')
else:
    training_losses = []
    eval_losses = []
    epoch = 0

print('Loading evaluation sample')
cache = {}
eval_batch = [ s[0][0].to("cuda") for s in sample_pickled_states('data/processed/', BATCH_SIZE, 1) ]
eval_batch = torch.stack(eval_batch).to("cuda")

print('Loading training sample')
sample = [ s[0][0].to("cuda") for s in sample_pickled_states('data/processed/', BATCH_SIZE, 1) ] 


MIN_EPOCHS_PER_NEW_TRAINING_SAMPLE = BATCH_SIZE * 2
dropout=DROPOUT
learning_rate=LEARNING_RATE
weight_decay=WEIGHT_DECAY
temperature=TEMPERATURE
MIN_EPOCHS_BEFORE_SAMPLING = MIN_EPOCHS_PER_NEW_TRAINING_SAMPLE * 2
min_loss = None
min_loss_epoch = None
hard=False
with ThreadPoolExecutor(max_workers=2) as executor:
    training_future = executor.submit(sample_pickled_states, 'data/processed/', BATCH_SIZE, 1)
    eval_future = executor.submit(sample_pickled_states, 'data/processed/', BATCH_SIZE, 1)

    train_epoch = 0
    new_eval = False
    frame_future = None
    is_sampling=False
    eval_index=0
    try:
        while True:
            if VARIATIONAL:
                if epoch > MIN_EPOCHS_BEFORE_SAMPLING:
                    if random.random() < 0.05:
                        if is_sampling:
                            print("Switching to mean mode")
                            model.set_mean_mode()
                            is_sampling = False
                        else:
                            print("Switching to sample mode")
                            model.set_sample_mode()
                            is_sampling = True
                    else:
                        model.set_mean_mode()
            # New training sample every X epocs
            if train_epoch >= MIN_EPOCHS_PER_NEW_TRAINING_SAMPLE and training_future.done():
                print(f'Loading new training sample')
                sample = [ s[0][0].to("cuda") for s in training_future.result() ]
                training_future = executor.submit(sample_pickled_states, 'data/processed/', BATCH_SIZE, 1)
                train_epoch = 0
            else:
                random.shuffle(sample)
                train_epoch += 1
            
            # New eval sample on request
            if new_eval:
                if eval_future.done():
                    print(f'Loading new eval sample')
                    eval_batch = [ s[0][0].to("cuda") for s in eval_future.result() ]
                    eval_batch = torch.stack(eval_batch).to("cuda")
                    new_eval = False
            epoch += 1

            # Train
            batch = torch.stack(sample).to("cuda")
            optimizer.zero_grad()
            output, _, _, _, _ = model(batch)
            loss = loss_fn(output, batch)
            loss.backward()
            optimizer.step()
            training_losses += [ loss.item() ]
            
            # Evaluate
            model.eval()
            eval_output, names_idxs, patterns, attributes_idxs, palettes = model(eval_batch)
            eval_loss = loss_fn(eval_output, eval_batch)
            eval_losses += [ eval_loss.item() ]
            model.train()
            
            if min_loss is None or eval_loss.item() < min_loss:
                min_loss = eval_loss.item()
                min_loss_epoch = epoch
            
            # Display (rounding to 6 decimals)
            print(f'Epoch {epoch}; loss: {round(loss.item(), 6)} (eval: {round(eval_loss.item(), 6)}); min loss: {round(min_loss, 6)} (epochs since min {epoch - min_loss_epoch})')
            if DISPLAY:
                if frame_future is None or frame_future.done():
                    if frame_future is not None:
                        _ = frame_future.result()
                    i = eval_index
                    frame_future = executor.submit(make_frame,
                                            stage,
                                            screen,
                                            eval_batch[i].cpu().detach().permute(2, 1, 0),
                                            eval_output[i].cpu().detach().permute(2, 1, 0),
                                            names_idxs[i].cpu().detach(),
                                            patterns[i].cpu().detach(),
                                            attributes_idxs[i].cpu().detach(),
                                            palettes[i].cpu().detach())

                # Handle events
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_d:
                            # Capital up, lowercase down
                            if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                                dropout += 0.1
                                if dropout > 1.0:
                                    dropout = 1.0
                            else:
                                dropout -= 0.1
                                if dropout <= 0.0:
                                    dropout = 0.0
                            print(f'Dropout: {dropout}')
                            set_dropout_rate(model, dropout)
                        elif event.key == pygame.K_l:
                            if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                                learning_rate *= 10
                            else:
                                learning_rate /= 10
                            print(f'Learning rate: {learning_rate}')
                            for param_group in optimizer.param_groups:
                                param_group['lr'] = learning_rate
                        elif event.key == pygame.K_r:
                            if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                                if weight_decay == 0:
                                    weight_decay = 1e-8
                                else:
                                    weight_decay *= 10
                            else:
                                weight_decay /= 10
                                if weight_decay < 1e-8:
                                    weight_decay = 0
                            print(f'Weight decay: {weight_decay}')
                            for param_group in optimizer.param_groups:
                                param_group['weight_decay'] = weight_decay
                        elif event.key == pygame.K_i:
                            init_weights(model)
                            reset_optimizer_state(optimizer)
                            epoch = 0
                            print('Weights reinitialized')
                        elif event.key == pygame.K_e:
                            print('preparing a new eval sample')
                            new_eval = True
                            eval_future = executor.submit(sample_pickled_states, 'data/processed/', BATCH_SIZE, 1)
                        elif event.key == pygame.K_s:
                            print('saving weights')
                            state = {
                                'model': model.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'epoch': epoch,
                                'training_losses': training_losses,
                                'eval_losses': eval_losses,
                            }
                            torch.save(state, 'data/training/base_ae.pth')
                        elif event.key == pygame.K_t:
                            if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                                temperature *= 2.0
                                if temperature >= 2.0:
                                    temperature = 2.0
                            else:
                                temperature /= 2.0
                                if temperature <= 0.001:
                                    temperature = 0.001
                            model.set_temperature(temperature, hard)
                            print(f'Temperature: {temperature}')
                        elif event.key == pygame.K_v:
                            if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                                eval_index += 1
                                if eval_index >= BATCH_SIZE:
                                    eval_index = 0
                            else:
                                eval_index -= 1
                                if eval_index < 0:
                                    eval_index = BATCH_SIZE - 1
                            print(f'Eval index: {eval_index}')
                        elif event.key == pygame.K_h:
                            if hard:
                                hard = False
                            else:
                                hard = True
                            model.set_temperature(temperature, hard)
                            print(f'Hard: {hard}')
                    
    except KeyboardInterrupt:
        pass