from base_agent import base_agent
from run_experiment import run_experiment
import torch
import torch.nn as nn
from torch.distributions import Categorical
import gym
import numpy as np

class PGN(nn.Module):    
    def __init__(self, input_size, n_actions):        
        super(PGN, self).__init__()        
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):        
        return self.net(x)


class reinforce_agent(base_agent):
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pgn = PGN(len(env.reset()), env.env.nA).to(device=device)
        self.alpha = learning_rate
        self.gamma = reward_decay

    def choose_action(self, s) -> int:
        logits = self.pgn(s)
        probs = Categorical(logits=logits) 
        a = probs.sample()
        return a

    def learn(self, s, a, r, s_, done) -> int:
        pass

if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    
    