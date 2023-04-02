# blipcon
#
# a framework for generating novel classic-style games
#
# There are three components to this project:
#
# 1. An agent RL AI that generates game state/action
#    pairs by playing instrumented games with an
#    open-ended novelty-based reward function.
#
# 2. A hybrid vae/transformer "writer" that functions
#    as a hypernetwork, generating the weights for
#    a "player" that predicts subsequent game states.
#    The writer produces a latent variational distribution
#    analogous to the output of a vae (the "cartridge")
#    from which the player's weights are sampled.
# 
# 3. A diffusion process that is trained to denoise
#    "cartridges" to produce a "clean" latent, with
#    loss weighted for the "playability" of the game
#    as measured by the amount of novelty the agent
#    can extract from the game.
#
# Note that 2 & 3 are the critical parts of the project,
# whereas 1 can be replaced with human feedback or more
# sophisticated agents/scoring algorithms.
#
# For the first cut, I will manually generate the training
# data for #2 by playing the game myself.

from typing import Optional

import math, random

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T

import cv2
import lz4.frame
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image

import pygame

from dist import MiniDist
from graphs import *
from positional import PositionalEncoding
from states import *
from video import *


# Writer network
class Writer(nn.Module):
    def __init__(self, embed_dim=64, num_heads=4, weights_out=8, input_dims=(240, 256)):
        super(Writer, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        )
        x, y = input_dims
        x, y = x // 4, y // 4
        feature_size = x * y * 64
        self.action_embedding = nn.Linear(8, feature_size)
        self.query = nn.Linear(feature_size, embed_dim)
        self.key = nn.Linear(feature_size, embed_dim)
        self.value = nn.Linear(feature_size, embed_dim)
        self.attention_layers = nn.MultiheadAttention(embed_dim, 4)
        self.positional_encoding = PositionalEncoding(feature_size)
        # For now, just do a dumb linear operation on the weights
        self.weight_generator = nn.Linear(embed_dim, weights_out)
        self.do_sample = False

    def forward(self, states, actions):
        batch_size, num_steps, _, _, _ = states.shape
        state_features = self.conv_layers(states.view(-1, 3, 240,  256))
        state_features = state_features.view(batch_size, num_steps, -1)

        action_features = self.action_embedding(actions.view(-1, 8))
        state_action_features = state_features + action_features
        state_action_features = self.positional_encoding(state_action_features)
        
        q, k, v = self.query(state_action_features), self.key(state_action_features), self.value(state_action_features)
        latents, _ = self.attention_layers(q, k, v)
        dist = MiniDist(latents)

        if self.do_sample:
            z = dist.sample()
        else:
            z = dist.mean
        
        player_weights = self.weight_generator(z)

        return player_weights, z
    
    def get_state_embedding(self, state, action=None):
        state_features = self.conv_layers(state)
        state_features = state_features.view(1, -1)
        if action is None:
            return state_features
        action_features = self.action_embedding(action)
        return state_features + action_features

# Player network
class Player(nn.Module):
    def __init__(self, output_dims=(240, 256)):
        super(Player, self).__init__()
        self.output_layers = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, output_padding=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, output_padding=1, padding=1),
            nn.Sigmoid()
        )
        self.output_dims = output_dims

    def get_weight_size(self):
        total = 0
        for p in self.parameters():
            total += p.view(-1).shape[0]
        return total

    def set_weights(self, weights):
        offset = 0
        for p in self.parameters():
            size = p.view(-1).shape[0]
            p.view(-1).data = weights[offset:offset + size]
            offset += size
        pass

    def forward(self, state_embedding):
        x, y = self.output_dims
        state_embedding = state_embedding.view(1, 64, x // 4, y // 4)
        next_state = self.output_layers(state_embedding)
        return next_state


DISPLAY = True
if DISPLAY:
    screen = pygame.display.set_mode((256 * 2, 240 * 2))
BATCH_SIZE = 64
MEM_SIZE = 1024
EMBED_DIM = 64
player = Player().to("cuda")
writer = Writer(weights_out=player.get_weight_size(), embed_dim=EMBED_DIM).to("cuda")

optimizer = torch.optim.Adam(writer.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()
losses = []
loop = 0

if DISPLAY:
    video_writer = VideoWriter("test.mp4", (256 * 2, 240 * 2))

try:
    while True:
        states, actions = [], []
        training_samples = []
        iteration = 0
        for img, buttons in load_states('states.lz4'):
            print(f"loop {loop}, iteration {iteration}, len(training_samples) {len(training_samples)}")
            iteration += 1

            states.append(img.to("cuda"))
            actions.append(buttons.to("cuda"))
            if len(states) < BATCH_SIZE:
                continue

            weights, latents = writer.forward(torch.stack(states).unsqueeze(0), torch.stack(actions).unsqueeze(0))
            player.set_weights(weights)

            total_loss = 0
            choices = set([ random.randrange(0, len(states) - 1) for _ in range(8) ])
            training_samples += [ (states[choice], actions[choice], states[choice + 1]) for choice in choices ]
            sampled_samples = random.sample(training_samples, min(len(training_samples), 8))
            last_state = (states[-2], actions[-2], states[-1])
            for i, (state, action, next_state) in enumerate([last_state] + sampled_samples):
                expected = player.forward(writer.get_state_embedding(state, action))
                optimizer.zero_grad()
                actual = next_state
                if DISPLAY:
                    if i == 0:
                        screen.blit(pil_to_surface(to_image(actual)), (0, 0))
                        screen.blit(pil_to_surface(to_image(expected.squeeze())), (256, 0))
                loss = loss_fn(expected.squeeze(), actual)
                loss.backward()
                total_loss += loss.item()
                optimizer.step()
            losses += [total_loss / len(training_samples)]

            # Plot the current losses use a canvas size of 512 x 240 pixels
            if DISPLAY:
                filtered = [ l for l in losses if l < 0.1 ]
                img1 = mk_pil_graph((256, 240), filtered, 'Loss over time', 'Iteration', 'Loss')
                screen.blit(pil_to_surface(img1), (0, 240))
                img2 = mk_pil_graph((256, 240), filtered[-100:], 'Recent Loss', 'Iteration', 'Loss')
                screen.blit(pil_to_surface(img2), (256, 240))
                pygame.display.flip()
                
                # Convert screen to cv2 video frame
                video_writer.frame_from_surface(screen)

            states.pop(0)
            actions.pop(0)
            while len(training_samples) > MEM_SIZE:
                training_samples.pop(0)
        loop += 1
except KeyboardInterrupt:
    pass

if DISPLAY:
    video_writer.close()
