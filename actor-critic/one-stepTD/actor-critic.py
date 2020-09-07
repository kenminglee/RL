from base_agent import base_agent
from run_experiment import run_cartpole_experiment
import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
import random
import torch.nn.functional as F
import gym

# Advantage actor-critic


class Critic(nn.Module):
    # Dueling architecture https://arxiv.org/pdf/1511.06581.pdf
    def __init__(self, input_size, n_actions):
        super(DQN, self).__init__()
        self.n_actions = n_actions
        self.fc1 = nn.Linear(input_size, 128)
        self.fc_state_action_val_output = nn.Linear(128, n_actions)
        self.fc_state_val_output = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        stream1_out = self.fc_state_action_val_output(x)
        stream2_out = self.fc_state_val_output(x)
        stream1_out_mean = torch.sum(
            stream1_out, dim=1, keepdim=True) / self.n_actions
        output = stream2_out + (stream1_out - stream1_out_mean)
        return output


class Actor(nn.Module):
    def __init__(self, input_size, n_actions):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        return self.net(x)


class agent(base_agent):
    def __init__(self, input_size, actions, learning_rate=0.001, reward_decay=0.9):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.actor = Actor(input_size, len(actions)
                           ).double().to(device=self.device)
        self.critic = Critic(input_size, len(
            actions)).double().to(device=self.device)
        

    def choose_action(self, s) -> int:
        pass

    def learn(self, s, a, r, s_, done) -> int:
        pass
