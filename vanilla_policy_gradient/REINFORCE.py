from base_agent import base_agent
from run_experiment import run_cartpole_experiment
import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.optim as optim
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
    def __init__(self, state_dim, actions, learning_rate=0.001, reward_decay=0.9):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.pgn = PGN(state_dim, len(actions)).to(device=self.device).double()
        self.gamma = reward_decay
        self.obs = []
        self.optimizer = optim.Adam(self.pgn.parameters(), lr=learning_rate)

    def choose_action(self, s) -> int:
        s = torch.tensor(s).to(device=self.device)
        logits = self.pgn(s)
        probs = Categorical(logits=logits)
        a = probs.sample()
        return a.tolist()

    def learn(self, s, a, r, s_, done) -> int:
        self.obs.append((s, a, r))
        if done:
            self.perform_learning_iter()
            self.obs = []
        return self.choose_action(s_)

    def perform_learning_iter(self):
        states, actions, rewards = zip(*self.obs)
        discounted_rewards = self.convert_to_discounted_reward(rewards)
        self.optimizer.zero_grad()
        states = torch.tensor(states, dtype=torch.double, device=self.device)
        actions = torch.tensor(actions, device=self.device)
        discounted_rewards = torch.tensor(
            discounted_rewards, device=self.device)
        m = Categorical(logits=self.pgn(states))
        loss = torch.sum(-m.log_prob(actions)*discounted_rewards)
        loss.backward()
        self.optimizer.step()

    def convert_to_discounted_reward(self, rewards):
        cumulative_reward = [rewards[-1]]
        for reward in reversed(rewards[:-1]):
            cumulative_reward.append(reward + self.gamma*cumulative_reward[-1])
        return cumulative_reward[::-1]

    def save_model(self, path):
        torch.save(self.pgn, path)


if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    agent = reinforce_agent(env.observation_space.shape[0], [
                            i for i in range(env.action_space.n)])
    run_cartpole_experiment(agent)
    agent.save_model('vanilla_policy_gradient.pt')
