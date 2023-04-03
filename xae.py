"""
An exploratory ae project to attempt to
"map" the distribution of the "game states."

The goal is to determine the best structure
for a state predictor capable of reproducing
the game states perfectly.
"""

import os, random

import torch
from torch import nn
from torchvision import transforms as T

import PIL
import pygame

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
                 hidden_size=75840):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(2),
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

# Collect the sample images
BATCH_SIZE = 64
to_image = T.ToTensor()
dir_ = 'samples_downsized'
imgs = []
for file in os.listdir(dir_):
    img = PIL.Image.open(os.path.join(dir_, file))
    imgs.append(to_image(img))
sample = [ s.to("cuda") for s in random.sample(imgs, BATCH_SIZE) ]
batch = torch.stack(sample).to("cuda")

model = BaseAE(hidden_size=75840 // 64).to("cuda")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

DISPLAY = True
if DISPLAY:
    screen = pygame.display.set_mode((512, 224))

# Try to load the weights
try:
    model.load_state_dict(torch.load('base_ae.pth'))
    print('Loaded weights')
except:
    print('No weights found')

EPOCHS = 1000
losses = []
try:
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        output = model(batch)
        loss = loss_fn(output, batch)
        loss.backward()
        optimizer.step()
        losses += [loss.item()]
        if epoch % 10 == 0:
            print(f'Epoch {epoch} loss: {loss.item()}')
            if DISPLAY:
                i = random.randrange(BATCH_SIZE)
                img1 = batch[i].cpu().detach().permute(2, 1, 0)
                img1 = img1.numpy() * 255
                img1 = img1.astype('uint8')
                img1 = pygame.surfarray.make_surface(img1)
                screen.blit(img1, (0, 0))
                img2 = output[i].cpu().detach().permute(2, 1, 0)
                img2 = img2.numpy() * 255
                img2 = img2.astype('uint8')
                img2 = pygame.surfarray.make_surface(img2)
                screen.blit(img2, (256, 0))
                pygame.display.flip()
except KeyboardInterrupt:
    pass

# Save the model weights
torch.save(model.state_dict(), 'base_ae.pth')