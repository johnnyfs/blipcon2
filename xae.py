"""
An exploratory ae project to attempt to
"map" the distribution of the "game states."

The goal is to determine the best structure
for a state predictor capable of reproducing
the game states perfectly.
"""

from concurrent.futures import ThreadPoolExecutor
import random

import torch
from torch import nn
from torchvision import transforms as T

import PIL
import pygame

from processing.states import sample_pickled_states
from modules.management import *
from modules.ae import MiniVae
from modules.doubleq import DoubleDQNAgent
from modules.nes import ResNetAE
from modules.enums import *
from visualization.images import pil_to_surface


def make_frame(stage, screen, actual, predicted,
               names_idxs=None, patterns=None, attributes_idxs=None, palettes=None):
    img1 = actual.numpy() * 255
    img1 = img1.astype('uint8')
    img1 = pygame.surfarray.make_surface(img1)
    stage.blit(img1, (0, 0))
    img2 = predicted.numpy() * 255
    img2 = img2.astype('uint8')
    img2 = pygame.surfarray.make_surface(img2)
    stage.blit(img2, (256, 0))
    
    if names_idxs is not None:
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
    
    if patterns is not None:
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
    
    if attributes_idxs is not None:
        # attributes are 4x15x16
        attributes_idxs = attributes_idxs.view(8, 15, 16).permute(2, 1, 0).argmax(dim=2)
        attributes_idxs = attributes_idxs.unsqueeze(-1).repeat_interleave(3, dim=-1)
        img5 = attributes_idxs.numpy().astype('uint8') * 36
        img5 = PIL.Image.fromarray(img5, mode='RGB')
        img5 = pil_to_surface(img5)
        img5 = pygame.transform.scale(img5, (32*8, 30*8))
        stage.blit(img5, (0, 240 + 256))
    
    if palettes is not None:
        # palettes are 4,4*3
        palettes = palettes.view(8, 4, 3)
        img6 = (palettes * 255).numpy().astype('uint8')
        img6 = PIL.Image.fromarray(img6, mode='RGB')
        img6 = pil_to_surface(img6)
        img6 = pygame.transform.scale(img6, (8*16, 4*16))
        stage.blit(img6, (256, 240 + 256))
    
    pygame.transform.scale(stage, (stage.get_width() * SCALE, stage.get_height() * SCALE), screen)
    pygame.display.flip()
              

BATCH_SIZE=1
DROPOUT=0.5
LEARNING_RATE=1e-3
WEIGHT_DECAY=1e-4
TEMPERATURE=2.0
ACT=NonLinearity.SILU

DISPLAY=True
SCALE=1
RELOAD=False
VARIATIONAL=False

if not VARIATIONAL:
    model: nn.Module = ResNetAE(
        input_channels=3,
        layers=[32, 32, 64, 64],
        dropout=DROPOUT,
        nonlinearity=ACT,
        layers_per_block=1,
        latent_channels=8,
        residual=1).to("cuda")
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
agent = DoubleDQNAgent(
    state_size=8 * 30 * 32,
    action_size=8,
    batch_size=1
)
q_mode = False

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
state = sample[0]

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
    if not q_mode:
        training_future = executor.submit(sample_pickled_states, 'data/processed/', BATCH_SIZE, 1)
        eval_future = executor.submit(sample_pickled_states, 'data/processed/', BATCH_SIZE, 1)

    train_epoch = 0
    new_eval = False
    frame_future = None
    is_sampling=False
    eval_index=0
    try:
        while True:
            if q_mode:
                print(epoch)
                model.eval()
                with torch.no_grad():
                    embedding = model.encode(state.unsqueeze(0)).flatten()
                agent.handle(embedding, 0, False)
                epoch += 1
                continue
                
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
            #output, _, _, _, _ = model(batch)
            output = model(batch)
            loss = loss_fn(output, batch)
            loss.backward()
            optimizer.step()
            training_losses += [ loss.item() ]
            
            # Evaluate
            model.eval()
            #eval_output, names_idxs, patterns, attributes_idxs, palettes = model(eval_batch)
            eval_output = model(eval_batch)
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
                                            eval_output[i].cpu().detach().permute(2, 1, 0))
                                            # names_idxs[i].cpu().detach(),
                                            # patterns[i].cpu().detach(),
                                            # attributes_idxs[i].cpu().detach(),
                                            # palettes[i].cpu().detach())

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
                            if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                                state = torch.load('data/training/base_ae.pth')
                                model.load_state_dict(state['model'])
                                optimizer.load_state_dict(state['optimizer'])
                                epoch = state['epoch']
                                training_losses = state['training_losses']
                                eval_losses = state['eval_losses']
                                min_loss = state.get('min_loss', min_loss)
                                min_loss_epoch = stage.get('min_loss_epoch', min_loss_epoch)
                            else:
                                print('saving weights')
                                state = {
                                    'model': model.state_dict(),
                                    'optimizer': optimizer.state_dict(),
                                    'epoch': epoch,
                                    'training_losses': training_losses,
                                    'eval_losses': eval_losses,
                                    'min_loss': min_loss,
                                    'min_loss_epoch': min_loss_epoch,
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
                        elif event.key == pygame.K_q:
                            q_mode = not q_mode
                            print(f'Q mode: {q_mode}')
                            if q_mode:
                                model.eval()
                            else:
                                model.train()
                    
    except KeyboardInterrupt:
        pass