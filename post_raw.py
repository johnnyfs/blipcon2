from processing.states import load_states
import requests
import torch

# Collect command line arguments infile, outfile
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('filename', help='lz4 compressed state file')
args = parser.parse_args()

print(f'Loading {args.filename}...')
for state, action, reward in load_states(args.filename, expect_reward=True):
    data = {
        'type': 'list',
        'state': state.tolist(),
        'action_override': action.tolist(),
        'reward': reward   
    }
    requests.post('http://localhost:5000/predict', json=data)