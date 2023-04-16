
from collections import deque
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_layers, seed):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        layers = [nn.Linear(state_size, hidden_layers[0])]
        layers += [nn.ReLU()]

        for i in range(len(hidden_layers) - 1):
            layers += [nn.Linear(hidden_layers[i], hidden_layers[i + 1])]
            layers += [nn.ReLU()]

        layers += [nn.Linear(hidden_layers[-1], action_size)]

        self.model = nn.Sequential(*layers)

    def forward(self, state):
        return self.model(state)

class DoubleDQNAgent:
    def __init__(self, state_size, action_size, hidden_layers=[64, 64], seed=0,
                 buffer_size=int(1e5), batch_size=64, gamma=0.99, tau=1e-3,
                 learning_rate=5e-4, update_every=4):
        
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        self.qnetwork_local = QNetwork(state_size, action_size, hidden_layers, seed).to('cuda')
        self.qnetwork_target = QNetwork(state_size, action_size, hidden_layers, seed).to('cuda')
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=learning_rate)

        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.update_every = update_every
        self.t_step = 0
        
        self.previous_state
        self.previous_action

    def step(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            if len(self.memory) > self.batch_size:
                experiences = self.sample()
                self.learn(experiences, self.gamma)

    def act(self, state, eps=0.):
        state = torch.from_numpy(state).float().unsqueeze(0).to('cuda')
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        action_probs = torch.sigmoid(action_values).cpu().data.numpy()
        action_thresholds = np.random.uniform(size=self.action_size)

        if random.random() > eps:
            return (action_probs > action_thresholds).astype(np.int)
        else:
            return np.random.randint(2, size=self.action_size)

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences

        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        Q_expected = self.qnetwork_local(states).gather(1, actions)

        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def sample(self, batch_size):
        if len(self.memory) < batch_size:
            return None

        # Randomly sample a batch of experiences from the memory
        experiences = random.sample(self.memory, batch_size)

        # Extract the elements of the experience tuple and stack them into separate tensors
        states = torch.stack([torch.tensor(exp.state, dtype=torch.float32) for exp in experiences]).to(self.device)
        actions = torch.stack([torch.tensor(exp.action, dtype=torch.long) for exp in experiences]).to(self.device)
        rewards = torch.stack([torch.tensor(exp.reward, dtype=torch.float32) for exp in experiences]).to(self.device)
        next_states = torch.stack([torch.tensor(exp.next_state, dtype=torch.float32) for exp in experiences]).to(self.device)
        dones = torch.stack([torch.tensor(exp.done, dtype=torch.bool) for exp in experiences]).to(self.device)

        return states, actions, rewards, next_states, dones
        
    def handle(self, state, reward, terminal):
        initial = self.previous_state == None
        
        self.previous_state = state
        self.previous_action = self.act(state)
        
        if not initial:
            self.step(self.previous_state, self.previous_action, state, reward, terminal)

        return self.previous_action
