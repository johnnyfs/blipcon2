import base64

from flask import Flask, request, jsonify

from objects.state import State
from modules.doubleq import DoubleQ

agent = DoubleQ()
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    state_str = base64.decodebytes(data['state'].encode('utf-8'))
    reward = float(base64.decodebytes(data['reward'].encode('utf-8')))

    state = State(state_str)

    action_probabilities = agent.handle(state, reward)

    # Convert tensor to a list of floats
    actions = action_probabilities.tolist()

    return jsonify({'actions': actions})

