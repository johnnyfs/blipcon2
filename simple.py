from typing import Optional

import math, random
import pickle

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

from blipcon import MiniDownBlock, MiniUpBlock
from graphs import *
from initializers import *
from positional import PositionalEncoding
from loss import combined_masked_loss
from states import *
from video import VideoWriter

class ResDown(nn.Module):
    def __init__(self, in_channels, out_channels, norm_groups, dropout=0.0):
        super(ResDown, self).__init__()
        self.norm = nn.GroupNorm(norm_groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        hidden = self.norm(x)
        hidden = self.relu(hidden)
        hidden = self.conv1(hidden)
        hidden = self.relu(hidden)
        hidden = self.dropout(hidden)

        x = self.conv2(x)
        x = x + hidden
        x = self.pool(x)

        return x
    
class ResUp(nn.Module):
    def __init__(self, in_channels, out_channels, norm_groups, dropout=0.0):
        super(ResUp, self).__init__()
        self.norm = nn.GroupNorm(norm_groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.trans = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        hidden = self.norm(x)
        hidden = self.relu(hidden)
        hidden = self.conv1(hidden)
        hidden = self.relu(hidden)
        hidden = self.dropout(hidden)

        x = self.conv2(x)
        x = x + hidden
        x = self.trans(x)

        return x


class StatePredictor(nn.Module):
    """
    Implements a simplified version of the above architecture:

    The network predicts the next state given a series of state-action pairs.

    States are passed through a series of convolutional layers.
    Actions are passed through a linear layer and embedded with the states.

    The resulting state-action features are passed through a series of
    multi-head attention layers.

    The final output is a linear layer that predicts the next state.
    The state is decoded from the latent representation using a series of
    convolutional transpose layers.

    Configured by
    - spatial_dims: dimensions of the input state
    - embed_dim: dimension of the latent space
    - num_heads: number of attention heads
    - conv_layers: channels per conv layer (encoding/decoding are symmetric)
    """
    def __init__(self,
                 spatial_dims=(240, 256),
                 input_channels=3,
                 embed_dim=64,
                 num_heads=4,
                 conv_layers=[32, 64, 64],
                 norm_groups=8,
                 dropout=0.0):
        super(StatePredictor, self).__init__()
        
        self.input_layers = nn.Sequential(
            nn.Conv2d(input_channels, conv_layers[0], kernel_size=3, stride=1, padding=1),
        )

        # Encoding state
        x, y = spatial_dims
        for i in range(1, len(conv_layers)):
            self.input_layers.add_module(f"down{i}", ResDown(conv_layers[i - 1], conv_layers[i], norm_groups=norm_groups, dropout=dropout))
            x, y = x // 2, y // 2

        self.features_dim = (y, x, conv_layers[-1])
        feature_size = x * y * conv_layers[-1]
        print("Feature size:", feature_size)

        # Build the action input layers
        self.action_embedding = nn.Linear(8, feature_size)

        # Build the transformer layers
        self.positional_encoding = PositionalEncoding(feature_size)
        self.query = nn.Linear(feature_size, embed_dim)
        self.key = nn.Linear(feature_size, embed_dim)
        self.value = nn.Linear(feature_size, embed_dim)
        self.attention_layers = nn.MultiheadAttention(embed_dim, num_heads)

        # Post-processing
        self.feedforward = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, embed_dim)),
            nn.Linear(embed_dim, feature_size),
            nn.ReLU(),
        )
        if dropout > 0.0:
            self.feedforward.add_module("dropout", nn.Dropout(dropout))

        # Decoding state
        self.output_layers = nn.Sequential()
        for i in range(len(conv_layers) - 1, 0, -1):
            self.output_layers.add_module(f"up{i}", ResUp(conv_layers[i], conv_layers[i - 1], norm_groups=norm_groups, dropout=dropout))
            x, y = x * 2, y * 2
        
        self.output_layers.add_module(f"norm", nn.GroupNorm(norm_groups, conv_layers[0]))
        self.output_layers.add_module(f"silu", nn.SiLU())
        self.output_layers.add_module(f"up", nn.Conv2d(conv_layers[0], input_channels, kernel_size=3, stride=1, padding=1)),

    def forward(self, states, actions):
        # Input states are: (batch_size, sequence_size, channels, height, width)
        batch_size, sequence_size, *_ = states.shape

        # Combine the batch and sequence dimensions -> (batch * seq, ch, h, w)
        states = states.view(-1, *states.shape[2:])
        actions = actions.view(-1, *actions.shape[2:])

        # State is convolved to (bach * seq, feature_size)
        # Actions are mapped to the same dimensions
        state_features = self.input_layers(states)
        action_features = self.action_embedding(actions)

        # Restore the batch dimensions so we can positionally encode
        # (batch * seq, feature_size) -> (batch, seq, feature_size)
        state_features = state_features.view(batch_size, sequence_size, -1)
        action_features = action_features.view(batch_size, sequence_size, -1)
        state_action_features = state_features + action_features
        state_action_features = self.positional_encoding(state_action_features)

        # Apply attention
        q = self.query(state_action_features)
        k = self.key(state_action_features)
        v = self.value(state_action_features)
        latents, _ = self.attention_layers(q, k, v)

        # Rehape to feature dimensions for restoration
        # And pool to reduce to a single state
        features = self.feedforward(latents)
        features = features.view(-1, *reversed(self.features_dim))
        next_states = self.output_layers(features)

        # Only one state per sequence now, so restore the batch dimension only
        next_states = next_states.view(batch_size, *next_states.shape[1:])

        return next_states
    
DISPLAY = True
SCALE = 1
if DISPLAY:
    if SCALE == 1:
        stage = pygame.display.set_mode((256 * 2, 240 * 2))
    else:
        screen = pygame.display.set_mode((256 * 2 * SCALE, 240 * 2 * SCALE))
        stage = pygame.Surface((256 * 2, 240 * 2))
    video_writer = VideoWriter("simple.mp4", (256 * 2, 240 * 2))
else:
    video_writer = None

BATCH_SIZE = 8
SEQUENCE_SIZE = 8
TOTAL_MIN_SIZE = BATCH_SIZE * SEQUENCE_SIZE + (1 if BATCH_SIZE == 1 else 0)
loss_fn = nn.MSELoss()
def step(states, actions, model, optimizer, start=None):
    state_batches = []
    action_batches = []
    actual_states = []

    # On the first loop, we always use the last SEQUENCE_SIZE  + 1 states
    # On subsequent loops, we use whatever was passed in.
    if start is None:
        start = len(states) - SEQUENCE_SIZE - 1
    stop = start + SEQUENCE_SIZE
    state_batches.append(torch.stack(states[start:stop]))
    action_batches.append(torch.stack(actions[start:stop]))
    actual_states.append(states[stop])

    # The remaining are random samples of length SEQUENCE_SIZE
    for _ in range(BATCH_SIZE - 1):
        start = random.randrange(0, len(states) - SEQUENCE_SIZE - 1)
        stop = start + SEQUENCE_SIZE
        state_batches.append(torch.stack(states[start:stop]))
        action_batches.append(torch.stack(actions[start:stop]))
        actual_states.append(states[stop])

    optimizer.zero_grad()
    states_in = torch.stack(state_batches)
    actions_in = torch.stack(action_batches)

    predicted_states = model.forward(states_in, actions_in)
    actual_states = torch.stack(actual_states).to("cuda")
    #loss = combined_masked_loss(predicted_states, actual_states, DIFF_THRESHOLD, DIFF_WEIGTH)
    loss = loss_fn(predicted_states, actual_states)
    loss.backward()
    optimizer.step()

    if DISPLAY:
        latest_actual = actual_states[0]
        stage.blit(pil_to_surface(to_image(latest_actual)), (0, 0))
        # Get the first batch of the predicted_states tensor
        latest_predicted = predicted_states[0]
        stage.blit(pil_to_surface(to_image(latest_predicted)), (256, 0))

        img1 = mk_pil_graph((256, 240), losses, 'Loss over time', 'Iteration', 'Loss')
        stage.blit(pil_to_surface(img1), (0, 240))
        img2 = mk_pil_graph((256, 240), losses[-100:], 'Recent Loss', 'Iteration', 'Loss')
        stage.blit(pil_to_surface(img2), (256, 240))
        if SCALE > 1:
            pygame.transform.scale(stage, (256 * 2 * SCALE, 240 * 2 * SCALE), screen)
        pygame.display.flip()

        video_writer.frame_from_surface(stage)

        for events in pygame.event.get():
            if events.type == pygame.KEYDOWN:
                if events.key == pygame.K_e:
                    model.eval()
                elif events.key == pygame.K_t:
                    model.train()

    return loss.item()

RELOAD = True
SAVE = True
DIFF_WEIGTH = 10.0
DIFF_THRESHOLD = 0.1
model = StatePredictor(
    spatial_dims=(240, 256),
    input_channels=3,
    conv_layers=[32, 64, 64],
    embed_dim=64,
    num_heads=4,
    norm_groups=4,
    dropout=0.2
).to("cuda")
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

did_reload = False
if RELOAD:
    try:
        model.load_state_dict(torch.load("model.pt"))
        with open('states.pkl', 'rb') as f:
            epoch, states, actions, losses, done_loading = pickle.load(f)
        print("Loaded model weights and state")
        did_reload = True
    except:
        print("load failure, starting from scratch")
if not did_reload:
    print("Initializing model weights")
    losses = []
    epoch = 0
    states = []
    actions = []
    done_loading = False
    init_weights(model)

try:
    print("Loading states from states.lz4")
    if not done_loading:
        for state, action in load_states('states.lz4'):
            states.append(state.to("cuda"))
            actions.append(action.to("cuda"))

            print(f"epoch {epoch}; loaded {len(states)}/{SEQUENCE_SIZE}*{BATCH_SIZE}={TOTAL_MIN_SIZE} minimum states, latest loss: {losses[-8:]}", end='\r')
            if len(states) < TOTAL_MIN_SIZE:
                continue
            
            loss = step(states, actions, model, optimizer)
            losses += [loss]
            epoch += 1
    print("Done loading states")
    done_loading = True
    start = 0
    loops = 0
    while True:
        print(f"epoch {epoch}; replaying sequence at {start} (loop {loops}), latest loss: {losses[-8:]}", end='\r')
        loss = step(epoch, states, actions, model, optimizer, start)
        losses += [loss]
        epoch += 1
        start += 1
        if start > len(states) - SEQUENCE_SIZE - 1:
            start = 0
        loops += 1
except KeyboardInterrupt:
    print("Keyboard interrupt")

if DISPLAY:
    print("closing video writer")
    video_writer.close()

# Save everything
if SAVE:
    print("saving states")
    with open('states.pkl', 'wb') as f:
        pickle.dump((epoch, states, actions, losses, done_loading), f)

    torch.save(model.state_dict(), 'model.pt')
