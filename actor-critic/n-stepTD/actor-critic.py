from base_agent import base_agent
from run_experiment import run_cartpole_experiment
import torch.nn as nn
import torch
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import random
import torch.nn.functional as F
import gym

# Advantage actor-critic


class Critic(nn.Module):
    def __init__(self, input_size):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x)


class Actor(nn.Module):
    def __init__(self, input_size, n_actions):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        return self.net(x)


class NStepACAgent(base_agent):
    def __init__(self, input_size, actions, n_step=3, learning_rate=0.001, reward_decay=0.9):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.actor = Actor(input_size, len(actions)
                           ).double().to(device=self.device)
        self.critic = Critic(input_size).double().to(device=self.device)
        self.gamma = reward_decay
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=learning_rate)
        self.n_step = n_step
        self.remainingSteps = n_step
        self.obs = []
        self.latest_log_prob = []

    def choose_action(self, s) -> int:
        s = torch.tensor(s, dtype=torch.double).to(device=self.device)
        logits = self.actor(s)
        probs = Categorical(logits=logits)
        a = probs.sample()
        assert len(self.latest_log_prob) == 0
        self.latest_log_prob.append(probs.log_prob(a))
        return a.tolist()

    def learn(self, s, a, r, s_, done) -> int:
        self.obs.append((s, self.latest_log_prob.pop(), a, r, s_))
        self.remainingSteps -= 1
        if self.remainingSteps <= 0:
            self.perform_learning_iter(done)
        if done:
            self.obs = []
            self.remainingSteps = self.n_step
            return 0
        return self.choose_action(s_)

    def perform_learning_iter(self, done):
        states, log_probs, actions, rewards, next_states = zip(*self.obs)
        discounted_rewards = self.convert_to_discounted_reward(rewards)
        curr_state_train = torch.tensor(
            states[0], dtype=torch.double).to(device=self.device)
        curr_action_train = torch.tensor(
            actions[0], dtype=torch.double).to(device=self.device)
        next_state_train = torch.tensor(
            next_states[-1], dtype=torch.double).to(device=self.device)
        if done:
            delta = discounted_rewards[0] - self.critic(curr_state_train)
        else:
            delta = discounted_rewards[0] + self.critic(
                next_state_train) - self.critic(curr_state_train)

        self.actor_optimizer.zero_grad()
        m = Categorical(logits=self.actor(curr_state_train))
        actor_loss = -m.log_prob(curr_action_train)*(delta.detach())
        actor_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss = delta**2
        critic_loss.backward()
        self.critic_optimizer.step()

    def convert_to_discounted_reward(self, rewards):
        cumulative_reward = [rewards[-1]]
        for reward in reversed(rewards[:-1]):
            cumulative_reward.append(reward + self.gamma*cumulative_reward[-1])
        return cumulative_reward[::-1]

    def save_model(self, path):
        torch.save(self.actor, path)


if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    agent = NStepACAgent(env.observation_space.shape[0], [
        i for i in range(env.action_space.n)])
    run_cartpole_experiment(agent)
    agent.save_model('vanilla_actor_critic.pt')
