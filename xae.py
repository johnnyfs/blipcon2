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
from modules.blocks import MiniMid, MiniDownBlock, MiniUpBlock
from modules.enums import *


class ResNetAE(nn.Module):
    def __init__(self,
                 input_channels=3,
                 layers=[32, 64, 64],
                 norm_groups=8,
                 nonlinearity=NonLinearity.RELU,
                 dropout=0.0,
                 residual=1.0):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, layers[0], 3, padding=1),
            get_module_for(nonlinearity),
        )
        for i in range(1, len(layers)):
            is_last = i == len(layers) - 1
            down = DownSample.AVG_POOL
            block = MiniDownBlock(layers[i-1],
                                  layers[i],
                                  norm_groups=norm_groups,
                                  num_layers=1,
                                  nonlinearity=nonlinearity,
                                  down_type=down,
                                  dropout=dropout,
                                  residual=residual)
            self.encoder.add_module(f'dblock{i}', block)
        self.mid = MiniMid(layers[-1],
                           norm_groups=norm_groups,
                           num_layers=2,
                           nonlinearity=nonlinearity,
                           dropout=dropout,
                           residual=residual)
        self.decoder = nn.Sequential()
        for i in range(len(layers) - 1, 0, -1):
            is_last = i == len(layers) - 1
            up = UpSample.TRANSPOSE
            block = MiniUpBlock(layers[i],
                                layers[i-1],
                                norm_groups=norm_groups,
                                num_layers=1,
                                nonlinearity=nonlinearity,
                                up_type=up,
                                dropout=dropout,
                                residual=residual)
            self.decoder.add_module(f'ublock{i}', block)
        self.decoder.add_module('final', nn.Conv2d(layers[0], input_channels, 3, padding=1))
        self.decoder.add_module('sigmoid', nn.Sigmoid())

    def forward(self, x):
        x = self.encoder(x)
        x = self.mid(x)
        x = self.decoder(x)
        return x


def make_frame(stage, screen, actual, predicted):
    img1 = actual.numpy() * 255
    img1 = img1.astype('uint8')
    img1 = pygame.surfarray.make_surface(img1)
    stage.blit(img1, (0, 0))
    img2 = predicted.numpy() * 255
    img2 = img2.astype('uint8')
    img2 = pygame.surfarray.make_surface(img2)
    stage.blit(img2, (256, 0))
    pygame.transform.scale(stage, (w * 2 * SCALE, h * SCALE), screen)
    pygame.display.flip()
                              

BATCH_SIZE = 16
DROPOUT=0.5
LEARNING_RATE=1e-3
WEIGHT_DECAY=1e-4

DISPLAY=True
SCALE=2
RELOAD=False

model: nn.Module = ResNetAE(dropout=DROPOUT).to("cuda")
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
loss_fn = nn.MSELoss()

if DISPLAY:
    w, h = 256, 240
    screen = pygame.display.set_mode((w * 2 * SCALE, 240 * SCALE))
    stage = pygame.Surface((w * 2, h))

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


MIN_EPOCHS_PER_NEW_TRAINING_SAMPLE = (BATCH_SIZE ** 2) // 2
dropout=DROPOUT
learning_rate=LEARNING_RATE
weight_decay=WEIGHT_DECAY
with ThreadPoolExecutor(max_workers=2) as executor:
    training_future = executor.submit(sample_pickled_states, 'data/processed/', BATCH_SIZE, 1)
    eval_future = executor.submit(sample_pickled_states, 'data/processed/', BATCH_SIZE, 1)

    train_epoch = 0
    new_eval = False
    frame_future = None
    try:
        while True:
            # New training sample every X epocs
            if train_epoch >= MIN_EPOCHS_PER_NEW_TRAINING_SAMPLE and training_future.done():
                print(f'Loading new training sample')
                sample = [ s[0][0].to("cuda") for s in training_future.result() ]
                training_future = executor.submit(sample_pickled_states, 'data/processed/', BATCH_SIZE, 1)
                train_epoch = 0
            else:
                random.shuffle(sample)
            
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
            output = model(batch)
            loss = loss_fn(output, batch)
            loss.backward()
            optimizer.step()
            training_losses += [ loss.item() ]
            
            # Evaluate
            model.eval()
            eval_output = model(eval_batch)
            eval_loss = loss_fn(eval_output, eval_batch)
            eval_losses += [ eval_loss.item() ]
            model.train()
            
            # Display
            print(f'Epoch {epoch} loss: {loss.item()} (eval: {eval_loss.item()}))')
            if DISPLAY:
                if frame_future is None or frame_future.done():
                    if frame_future is not None:
                        _ = frame_future.result()
                    i = random.randrange(BATCH_SIZE)
                    frame_future = executor.submit(make_frame,
                                            stage,
                                            screen,
                                            eval_batch[i].cpu().detach().permute(2, 1, 0),
                                            eval_output[i].cpu().detach().permute(2, 1, 0))

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
                    
    except KeyboardInterrupt:
        pass