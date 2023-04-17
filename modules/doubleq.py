
from collections import deque
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

from modules.enums import NonLinearity, get_module_for
from modules.encoding import PositionalEncoding

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

 
class QNetworkHybrid(nn.Module):
    """
    A hybrid QNetwork/Transformer that works with sequences of positionally encoded
    states w/ self-attention.
    """
    def __init__(self,
                 state_size,
                 action_size,
                 target_size,
                 num_haeds=8,
                 dropout=0.0,
                 nonlinearity: NonLinearity=NonLinearity.RELU):
        super(QNetworkHybrid, self).__init__()
        self.positional = PositionalEncoding(state_size)
        self.action_embedding = nn.Linear(action_size, state_size)
        self.nonlinearity = get_module_for(nonlinearity)
        self.q = nn.Linear(state_size, target_size)
        self.k = nn.Linear(state_size, target_size)
        self.v = nn.Linear(state_size, target_size)
        self.attention = nn.MultiheadAttention(target_size,
                                               num_heads=num_haeds,
                                               dropout=dropout)
        self.decode = nn.Sequential(
            nn.AdaptiveAvgPool1d(target_size),
            nn.Linear(target_size, action_size)
        )
        self.target_size = target_size
        
    def forward(self, state, past_states=None, past_actions=None):
        b, s = state.shape

        # Embed the actions in the states
        # First adding a dummy action for the last state
        if past_states is not None:
            _, z, a = past_actions.shape
            # # Push the length into the batch dimension
            past_states = past_states.reshape(-1, s)
            past_actions = past_actions.reshape(-1, a)
            states = torch.cat([past_states, state.view(b, s)], dim=0)
            actions = torch.cat([past_actions, torch.zeros(b, a).to(past_actions.device)], dim=0)
            embedded = self.action_embedding(actions)
            # Restore the batch dimenson (with the new length)
            states = states.reshape(b, z + 1, s)
            embedded = embedded.reshape(b, z + 1, s)
            encoded = self.positional(states + embedded)
        else:
            encoded = self.positional(state.view(b, 1, s))
        
        # Apply attention
        q = self.q(encoded)
        k = self.k(encoded)
        v = self.v(encoded)
        x = self.attention(q, k, v)[0]
        
        # Result should be a single next action
        return self.decode(x.reshape(b, -1))


class Moment:
    def __init__(self, state, action, reward, next_state, done, recent_past_states, recent_past_actions):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done
        self.recent_past_states = recent_past_states
        self.recent_past_actions = recent_past_actions
    

class DoubleDQNAgent:
    def __init__(self, state_size, action_size, hidden_layers=[64, 64], seed=0,
                 buffer_size=int(1e5), batch_size=4, gamma=0.99, tau=1e-3,
                 learning_rate=5e-4, update_every=4, device='cuda', exclusive_actions=False,
                 eps=1.0, eps_decay=0.995, eps_min=0.01, short_term_memory_size=8,
                 dropout=0.0, nonlinearity: NonLinearity=NonLinearity.RELU):
        
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # self.qnetwork_local = QNetwork(state_size, action_size, hidden_layers, seed).to('cuda')
        # self.qnetwork_target = QNetwork(state_size, action_size, hidden_layers, seed).to('cuda')
        self.qnetwork_local = QNetworkHybrid(state_size,
                                             action_size,
                                             hidden_layers[0],
                                             dropout=dropout,
                                             nonlinearity=nonlinearity).to('cuda')
        self.qnetwork_target = QNetworkHybrid(state_size,
                                              action_size,
                                              hidden_layers[0],
                                              dropout=dropout,
                                              nonlinearity=nonlinearity).to('cuda')
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=learning_rate)

        self.memory = deque(maxlen=buffer_size)
        self.recent_past_states = deque(maxlen=short_term_memory_size)
        self.recent_past_actions = deque(maxlen=short_term_memory_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.update_every = update_every
        self.t_step = 0
        
        self.previous_state = None
        self.previous_action = None
        
        self.device = device
        self.exclusive_actions = exclusive_actions
        self.eps = eps
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        
        self.losses = []

    def step(self, state, action, reward, next_state, done,
             recent_past_states=None, recent_past_actions=None):
        self.memory.append(Moment(state, action, reward, next_state, done,
                                  recent_past_states, recent_past_actions))

        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            if len(self.memory) > self.batch_size:
                experiences = self.sample(self.batch_size)
                self.learn(experiences, self.gamma)
                for exp in experiences:
                    del exp
                torch.cuda.empty_cache()

    def act(self, state, eps=0.):
        state = state.unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            if len(self.recent_past_states) == 0:
                past_states = None
                past_actions = None
            else:
                past_states = torch.stack([s for s in self.recent_past_states]).view(1, -1, self.state_size).to(self.device)
                past_actions = torch.stack([a for a in self.recent_past_actions]).view(1, -1, self.action_size).to(self.device)
            action_values = self.qnetwork_local(state, past_states, past_actions)
        self.qnetwork_local.train()

        action_probs = torch.sigmoid(action_values).cpu()
        action_thresholds = torch.rand_like(action_probs).cpu()

        if random.random() > eps:
            action = (action_probs > action_thresholds).squeeze(0).type(torch.int64)
        else:
            action = torch.randint(0, 2, (self.action_size,)).squeeze(0).type(torch.int64)
        self.recent_past_states.append(state.squeeze(0))
        self.recent_past_actions.append(action)

        return action

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones, rpss, rpas = experiences
        rewards = rewards.unsqueeze(1)
        dones = dones.unsqueeze(1)

        if self.exclusive_actions:
            Q_targets_next = self.qnetwork_target(next_states, rpss, rpas).detach().max(1)[0].unsqueeze(1)
            Q_targets = rewards / self.action_size + (gamma * Q_targets_next * (1 - dones))
            Q_expected = self.qnetwork_local(states).gather(1, actions).max(1)[0].unsqueeze(1)
        else:
            Q_targets_next = self.qnetwork_target(next_states, rpss, rpas)
            Q_targets = rewards.repeat(1, self.action_size) + (gamma * Q_targets_next * (1 - dones.repeat(1, self.action_size)))
            Q_expected = self.qnetwork_local(states).gather(1, actions)

        loss = F.mse_loss(Q_expected, Q_targets)
        self.losses.append(loss.item())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)
        if self.eps > self.eps_min:
            self.eps *= self.eps_decay

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def sample(self, batch_size):
        if len(self.memory) < batch_size:
            return None

        # Randomly sample a batch of experiences from the memory
        experiences = random.sample(self.memory, batch_size)

        # Extract the elements of the experience tuple and stack them into separate tensors
        states = torch.stack([exp.state for exp in experiences]).to(self.device)
        actions = torch.stack([exp.action for exp in experiences]).to(self.device)
        rewards = torch.stack([torch.tensor(exp.reward) for exp in experiences]).to(self.device)
        next_states = torch.stack([exp.next_state for exp in experiences]).to(self.device)
        dones = torch.stack([torch.tensor(float(exp.done)) for exp in experiences]).to(self.device)
        recent_past_states = torch.stack([torch.stack([rps for rps in exp.recent_past_states]) for exp in experiences]).to(self.device)
        recent_past_actions = torch.stack([torch.stack([rpa for rpa in exp.recent_past_actions]) for exp in experiences]).to(self.device)
        return states, actions, rewards, next_states, dones, recent_past_states, recent_past_actions
        
    def handle(self, state, reward, terminal, action_override=None):
        initial = self.previous_state == None
        
        action = self.act(state, self.eps)
        if action_override is not None:
            action = torch.tensor(action_override).type(torch.int64)
        
        self.previous_state = state
        self.previous_action = action
        
        if not initial and len(self.recent_past_states) >= self.recent_past_states.maxlen:
            self.step(self.previous_state, self.previous_action, reward, state, terminal,
                      self.recent_past_states.copy(), self.recent_past_actions.copy())

        return self.previous_action
