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
from modules.resnets import MiniDownBlock, MiniUpBlock, DownSampleType, UpSampleType, NonLinearity

class BaseAE(nn.Module):
    """
    Let's begin with a simple linear AE.

    We know that interally the entire visual
    state can be represented by 9480 bytes.

    4096 x 2    = 8192 chr pals
    4 x 4 x 6/8 =   12 color pals
    32 x 30     =  960 chr indexes 
    16 x 15 / 4 =   60 palette indexes
    64 * ~4     =  256 sprite desc
                = 9480 * 8 = 75840 bits

    This implies a hidden layer with 75840 nodes?

    Of course, this assumes that the state is
    fully represented by the visual state.
    """
    def __init__(self,
                 dims=(224, 256),
                 input_channels=3,
                 hidden_size=75840,
                 dropout=0.0):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.ReLU(),

            nn.GroupNorm(8, 32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.AvgPool2d(2),

            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.AvgPool2d(2),
        )
        self.linear1 = nn.Linear(64 * dims[0] // 4 * dims[1] // 4, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 64 * dims[0] // 4 * dims[1] // 4)
        self.decoder = nn.Sequential(
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 2, stride=2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv2d(32, input_channels, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        batch, _, h, w = x.size()
        x = self.encoder(x)
        x = x.view(batch, -1)
        x = self.linear1(x)
        x = self.linear2(x)
        x = x.view(batch, 64, h // 4, w // 4)
        x = self.decoder(x)
        return x


class ResNetAE(nn.Module):
    def __init__(self,
                 dims=(224, 256),
                 input_channels=3,
                 layers=[32, 64, 64],
                 hidden_size=75840,
                 norm_groups=8,
                 dropout=0.0,
                 residual=1.0):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, layers[0], 3, padding=1),
            nn.ReLU(),
        )
        for i in range(1, len(layers)):
            is_last = i == len(layers) - 1
            down = DownSampleType.AVG_POOL
            block = MiniDownBlock(layers[i-1], layers[i], norm_groups=norm_groups, dropout=dropout, residual=residual, down_type=down)
            self.encoder.add_module(f'dblock{i}', block)
        h, w = dims[0] // 2 ** (len(layers) - 1), dims[1] // 2 ** (len(layers) - 1)
        self.feature_dims = (h, w)
        feature_size = layers[-1] * w * h
        self.linear1 = nn.Linear(feature_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, feature_size)
        self.decoder = nn.Sequential()
        for i in range(len(layers) - 1, 0, -1):
            is_last = i == len(layers) - 1
            up = UpSampleType.TRANSPOSE
            block = MiniUpBlock(layers[i], layers[i-1], norm_groups=norm_groups, dropout=dropout, residual=residual, up_type=up)
            self.decoder.add_module(f'ublock{i}', block)
        self.decoder.add_module('final', nn.Conv2d(layers[0], input_channels, 3, padding=1))
        self.decoder.add_module('sigmoid', nn.Sigmoid())

    def forward(self, x):
        batch, _, h, w = x.size()
        x = self.encoder(x)
        x = x.view(batch, -1)
        x = self.linear1(x)
        x = self.linear2(x)
        h, w = self.feature_dims
        x = x.view(batch, -1, h, w)
        x = self.decoder(x)
        return x
                                    


class RecodeAE(nn.Module):
    """
    Similar to the above, but the target of decoding is
    the chr tables, bg maps, and sprite data.

    - 2 sets of 256-chr 8x8 4-color tiles
    - 2 sets of 4 4-color palettes
    - 1 32x30 tile map
    - scroll offsets (2 0-7 values)
    - 1 16x15 palette selection map
    - one flag saying whether sprites are 8x8 or 8x16
    - 64 sprite descriptors
        - x, y location (1 byte each)
        - tile index (1 byte)
        - palette index 0-15
        - flipped H/V? behind background?
    """
    def __init__(self,
                 dims=(224, 256),
                 input_channels=3,
                 hidden_size=75840):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.GroupNorm(4, 32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.GroupNorm(4, 64),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(2),
        )
        self.linear1 = nn.Linear(64 * dims[0] // 4 * dims[1] // 4, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 64 * dims[0] // 4 * dims[1] // 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, input_channels, 3, padding=1),
        )

    def forward(self, x):
        batch, _, h, w = x.size()
        x = self.encoder(x)
        x = x.view(batch, -1)
        x = self.linear1(x)
        x = self.linear2(x)
        x = x.view(batch, 64, h // 4, w // 4)
        x = self.decoder(x)
        return x
    

# Collect the sample images
BATCH_SIZE = 16
EVAL_SETS = 1
to_image = T.ToTensor()
dir_ = 'samples_downsized'
imgs = []
# for file in os.listdir(dir_):
#     img = PIL.Image.open(os.path.join(dir_, file))
#     imgs.append(to_image(img))
# sample = [ s.to("cuda") for s in random.sample(imgs, BATCH_SIZE) ]
# batch = torch.stack(sample).to("cuda")

DROPOUT=0.5
LEARNING_RATE=1e-5
WEIGHT_DECAY=1e-6

model = ResNetAE(
    dims=(240, 256),
    hidden_size=1024,
    dropout=DROPOUT).to("cuda")
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
loss_fn = nn.MSELoss()

DISPLAY = True
if DISPLAY:
    screen = pygame.display.set_mode((512, 240))

# Try to load the weights
RELOAD = False
SAVE = True
training_losses = []
eval_losses = []
epoch = 0
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

print('Loading evaluation sample')
cache = {}
eval_batch = [ s[0][0] for s in sample_pickled_states('data/processed/', BATCH_SIZE, 1, cache=cache, max_cache_size=64) ]
eval_batch = torch.stack(eval_batch).to("cuda")

print('Loading training sample')
sample = [ s[0][0] for s in sample_pickled_states('data/processed/', BATCH_SIZE, 1, cache=cache, max_cache_size=64) ] 

NEW_EVAL_EVERY = 256
dropout=DROPOUT
learning_rate=LEARNING_RATE
weight_decay=WEIGHT_DECAY
with ThreadPoolExecutor(max_workers=2) as executor:
    training_future = executor.submit(sample_pickled_states, 'data/processed/', BATCH_SIZE, 1, cache=cache, max_cache_size=64)
    eval_future = executor.submit(sample_pickled_states, 'data/processed/', BATCH_SIZE, 1, cache=cache, max_cache_size=64)

    eval_epoch = 0
    try:
        while True:
            if training_future.done():
                print(f'Loading new training sample')
                sample = [s[0][0] for s in training_future.result()]
                training_future = executor.submit(sample_pickled_states, 'data/processed/', BATCH_SIZE, 1, cache=cache, max_cache_size=64)
            else:
                random.shuffle(sample)
            eval_epoch += 1
            if eval_epoch > NEW_EVAL_EVERY:
                if eval_future.done():
                    print(f'Loading new eval sample')
                    eval_batch = [s[0][0] for s in eval_future.result()]
                    eval_batch = torch.stack(eval_batch).to("cuda")
                    eval_epoch = 0
                    eval_future = executor.submit(sample_pickled_states, 'data/processed/', BATCH_SIZE, 1, cache=cache, max_cache_size=64)
            epoch += 1

            batch = torch.stack(sample).to("cuda")
            optimizer.zero_grad()
            output = model(batch)
            loss = loss_fn(output, batch)
            loss.backward()
            optimizer.step()
            training_losses += [ loss.item() ]
            model.eval()
            eval_output = model(eval_batch)
            eval_loss = loss_fn(eval_output, eval_batch)
            eval_losses += [ eval_loss.item() ]
            model.train()
            print(f'Epoch {epoch} loss: {loss.item()} (eval: {eval_loss.item()}))')
            if DISPLAY:
                i = random.randrange(BATCH_SIZE)
                img1 = eval_batch[i].cpu().detach().permute(2, 1, 0)
                img1 = img1.numpy() * 255
                img1 = img1.astype('uint8')
                img1 = pygame.surfarray.make_surface(img1)
                screen.blit(img1, (0, 0))
                img2 = eval_output[i].cpu().detach().permute(2, 1, 0)
                img2 = img2.numpy() * 255
                img2 = img2.astype('uint8')
                img2 = pygame.surfarray.make_surface(img2)
                screen.blit(img2, (256, 0))
                pygame.display.flip()

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
                    
    except KeyboardInterrupt:
        pass

    # Save the model weights
    if SAVE:
        print('saving weights')
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'training_losses': training_losses,
            'eval_losses': eval_losses,
        }
        torch.save(state, 'data/training/base_ae.pth')
