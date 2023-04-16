from flask import Flask, request, jsonify
from objects.state import State
from objects.action import Action
from agents.doubleq import DoubleQ

agent = DoubleQ()
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    state_str = data['state']
    action_str = data['action']
    reward = data['reward']

    state = State(state_str)
    action = Action(action_str)

    action_probabilities = agent.handle_frame(state, action, reward)

    # Convert tensor to a list of floats
    actions = action_probabilities.tolist()

    return jsonify({'actions': actions})

