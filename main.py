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

import math

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T

import lz4.frame
import matplotlib.pyplot as plt
import PIL.Image

import pygame

to_tensor = T.ToTensor()
to_image = T.ToPILImage()


# Load an lz4 compressed state file
def load_states(filename):
    with lz4.frame.open(filename, 'r') as f:
        # The first full line is the image state as text
        prev_gdstr = None
        prev_buttons = None
        s = 0
        while True:
            header = f.read(11)
            if header is None:
                return
            gdstr = f.read(256 * 240 * 4)
            if gdstr is None or len(gdstr) != 256 * 240 * 4:
                return
            buttons = f.read(8)
            if buttons is None or len(buttons) != 8:
                return

            # Poor man's compression
            if gdstr == prev_gdstr and buttons == prev_buttons:
                yield None

            # Images are stored as bytes ARGB, where A is always 0
            PIL.Image.frombytes
            img = PIL.Image.new('RGB', (256, 240))
            i = 0
            for y in range(240):
                for x in range(256):
                    img.putpixel((x, y), (gdstr[i + 1], gdstr[i + 2], gdstr[i + 3]))
                    i += 4
            state = to_tensor(img)
            s += 1

            # Action is button states as text "ABsSUDLR"
            # (where s = select and S = start)
            action = torch.tensor([float(x) for x in buttons])

            yield (state, action)


def pil_to_surface(pil):
    return pygame.image.fromstring(pil.tobytes(), pil.size, pil.mode).convert()


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1024):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model

        # Create a positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x * math.sqrt(self.d_model)
        x = x + self.pe[:, :x.size(1)]
        return x

class MiniDist:
    def __init__(self, params):
        self.params = params
        # Split the latent channels into mean and log variance
        mean, logvar = torch.chunk(params, 2, dim=1)
        self.mean = mean
        # Prevent numerical instability
        self.logvar = torch.clamp(logvar, min=-10, max=10)
        self.std = torch.exp(0.5 * logvar)

    def sample(self, generator: Optional[torch.Generator]=None):
        sample = torch.randn(self.mean.shape, device=self.params.device, generator=generator)
        return self.mean + self.std * sample

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

screen = pygame.display.set_mode((256 * 2, 240))
BATCH_SIZE = 64
EMBED_DIM = 64
player = Player()
writer = Writer(weights_out=player.get_weight_size(), embed_dim=EMBED_DIM)
states, actions = [], []
iteration = 0
optimizer = torch.optim.Adam(writer.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()
for img, buttons in load_states('blipcon2/states.lz4'):
    print(f"iteration {iteration}")
    iteration += 1

    states.append(img)
    actions.append(buttons)
    if len(states) < BATCH_SIZE:
        continue

    weights, latents = writer.forward(torch.stack(states).unsqueeze(0), torch.stack(actions).unsqueeze(0))
    player.set_weights(weights)

    for i in range(len(states) - 1):
        expected = player.forward(writer.get_state_embedding(states[i], actions[i]))
        optimizer.zero_grad()
        actual = states[i + 1]
        screen.blit(pil_to_surface(to_image(actual.squeeze())), (0, 0))
        screen.blit(pil_to_surface(to_image(expected.squeeze())), (256, 0))
        pygame.display.flip()
        loss = loss_fn(expected, actual)
        loss.backward()
        print(loss.item())
        optimizer.step()

    states.pop(0)

