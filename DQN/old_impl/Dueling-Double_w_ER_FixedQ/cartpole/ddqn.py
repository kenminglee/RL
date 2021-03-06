# vanilla DQN with experience replay and fixed q-targets (https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)
from base_agent import base_agent
from run_experiment import run_cartpole_experiment
import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
import random
import torch.nn.functional as F
import gym


class DQN(nn.Module):
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


class dqn_agent(base_agent):
    def __init__(self, state_dim, actions, experience_replay_size=10000, e_greedy=0.1, learning_rate=0.001, reward_decay=0.9):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.actions = actions
        self.epsilon = e_greedy
        self.gamma = reward_decay
        # frozen param - our target dqn does not learn
        self.dqn_target = DQN(state_dim, len(actions)).to(
            device=self.device).double()
        self.dqn_policy = DQN(state_dim, len(actions)).to(
            device=self.device).double()
        self.experience_replay = []
        self.experience_replay_capacity = experience_replay_size
        self.optimizer = optim.Adam(
            self.dqn_policy.parameters(), lr=learning_rate)

    def choose_action(self, s) -> int:
        if np.random.uniform() >= self.epsilon:
            # we don't plan to do any backprop during inference, so turn it off to convserve mem
            with torch.no_grad():
                action_values = self.dqn_policy(torch.tensor(
                    s, dtype=torch.double, device=self.device).unsqueeze(0))
            action = int(torch.argmax(action_values).item())
        else:
            action = np.random.choice(self.actions)
        return action

    def learn(self, s, a, r, s_, done) -> int:
        if len(self.experience_replay) >= self.experience_replay_capacity:
            self.experience_replay.pop(0)
        self.experience_replay.append((s, a, r, s_))
        if done:
            self.experience_replay[-1] = (s, a, r, None)
            self.perform_learning_iter()
        return self.choose_action(s_)

    def perform_learning_iter(self):
        random.shuffle(self.experience_replay)
        sample = self.experience_replay[-200:]
        states, actions, rewards, next_states = zip(*sample)

        next_states_non_terminal_mask = torch.tensor(
            [i is not None for i in next_states], dtype=torch.bool).to(device=self.device)
        states = torch.tensor(states, dtype=torch.double).to(
            device=self.device)
        actions = torch.tensor(actions).to(device=self.device)
        rewards = torch.tensor(rewards).double().to(device=self.device)
        non_terminal_next_states = torch.tensor(
            [i for i in next_states if i is not None], dtype=torch.double).to(device=self.device)

        # [1,2,3] (size=3) -> unsqueeze(1) -> [[1], [2], [3]] (size=3x1)
        # [[1], [2], [3]] (size=3x1) -> squeeze(1) -> [1,2,3]
        # gather() basically take the value of the actions we have taken
        predicted_values = self.dqn_policy(states).gather(
            1, actions.unsqueeze(1)).squeeze(1)

        next_states_values = torch.zeros(
            rewards.size(), dtype=torch.double).to(device=self.device)
        # make sure that dqn_target's param are frozen
        with torch.no_grad():
            # We are doing double DQN here!
            next_states_actions_values = self.dqn_policy(
                non_terminal_next_states)
            max_next_states_actions = torch.argmax(
                next_states_actions_values, dim=1)
            # if we can take two actions, it returns [[#, #],[#, #],...]
            # we take max(1) means take the max value for each [#, #]
            # everything that is not covered by the mask will be defaulted to 0
            next_states_values[next_states_non_terminal_mask] = self.dqn_target(
                non_terminal_next_states).gather(1, max_next_states_actions.unsqueeze(1)).squeeze(1)

        actual_values = rewards + self.gamma*next_states_values

        loss = F.mse_loss(predicted_values, actual_values)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient Clipping
        for param in self.dqn_policy.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        self.dqn_target.load_state_dict(self.dqn_policy.state_dict())

    def save_model(self, path):
        torch.save(self.dqn_policy, path)


if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    agent = dqn_agent(env.observation_space.shape[0], [
                      i for i in range(env.action_space.n)])
    run_cartpole_experiment(agent)
    agent.save_model('ddqn.pt')
