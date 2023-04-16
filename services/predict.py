import base64

from flask import Flask, request, jsonify

from processing.states import from_raw_state
from modules.doubleq import DoubleDQNAgent
from modules.enums import NonLinearity
from modules.nes import ResNetAE

agent = DoubleDQNAgent(
    state_size=3 * 256 * 240,
    action_size=8
)
ae = ResNetAE(
    input_channels=3,
    layers=[32, 32, 64, 64],
    norm_groups=4,
    nonlinearity=NonLinearity.SILU,
    dropout=0.0,
    residual=1.0,
    latent_channels=8,
    layers_per_block=1
)
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    state_str = base64.decodebytes(data['state'].encode('utf-8'))
    reward = float(base64.decodebytes(data['reward'].encode('utf-8')))

    state = from_raw_state(state_str)

    action_probabilities = agent.handle(state, reward)

    # Convert tensor to a list of floats
    actions = action_probabilities.tolist()

    return jsonify({'actions': actions})

