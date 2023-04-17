import base64

from flask import Flask, request, jsonify
import pygame
import torch

from processing.states import from_raw_state
from modules.doubleq import DoubleDQNAgent
from modules.enums import NonLinearity
from modules.nes import ResNetAE
from visualization.graphs import mk_pil_graph
from visualization.images import pil_to_surface

DISPLAY=True

print('Creating agent...')
agent = DoubleDQNAgent(
    state_size=8 * 30 * 32,
    action_size=8,
    learning_rate=1e-3,
    gamma=0.80,
    batch_size=64,
    short_term_memory_size=32,
    dropout=0.1
)
print('Creating autoencoder...')
ae = ResNetAE(
        input_channels=3,
        layers=[32, 32, 64, 64],
        dropout=0.0,
        nonlinearity=NonLinearity.SILU,
        layers_per_block=1,
        latent_channels=8,
        residual=1
).to('cuda')
print('Loading weights...')
state = torch.load('data/training/base_ae.pth')
ae.load_state_dict(state['model'])
ae.eval()
app = Flask(__name__)

if DISPLAY==True:
    screen = pygame.display.set_mode((256, 240 * 2))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    type_ = data.get('type', 'raw')
    
    if type_ == 'raw':
        state_str = base64.decodebytes(data['state'].encode('utf-8'))
        reward = float(base64.decodebytes(data['reward'].encode('utf-8')))
        action_override = data.get('action_override', None)
        if action_override is not None:
            action_override = base64.decodebytes(action_override.encode('utf-8'))
            action_override = [int(ch) for ch in action_override]
        state = from_raw_state(state_str).unsqueeze(0).to('cuda')
    elif type_ == 'list':
        state = torch.tensor(data['state']).unsqueeze(0).to('cuda')
        reward = data.get('reward', 0.0)
        action_override = data.get('action_override', None)
        if action_override is not None:
            action_override = torch.tensor(action_override).unsqueeze(0).to('cuda')
        print(f'state {state.shape}; reward {reward}; action_override: {action_override}')
    with torch.no_grad():
        encoding = ae.encode(state)
        reconstruction = ae.decode(encoding)
        print(reconstruction.shape)
        if DISPLAY:
            img = reconstruction.detach().cpu().squeeze(0).permute(2, 1, 0).numpy() * 255
            img = img.astype('uint8')
            img = pygame.surfarray.make_surface(img)
            screen.blit(img, (0, 0))
            
            img = mk_pil_graph((256, 240), agent.losses, 'Losses', 'Step', 'Loss')
            img = pil_to_surface(img)
            screen.blit(img, (0, 240))
            pygame.display.flip()
            
        embedding = encoding.flatten()

    # TODO: capture terminal state (is there such a thing?)
    action_probabilities = agent.handle(embedding, reward, terminal=False, action_override=action_override)

    # Convert tensor to a list of floats
    actions = action_probabilities.tolist()
    print(f'Actions: {actions}')

    return jsonify({'actions': actions})

