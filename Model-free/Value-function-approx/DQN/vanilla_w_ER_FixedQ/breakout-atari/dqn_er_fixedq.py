# vanilla DQN with experience replay and fixed q-targets (https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)
from base_agent import base_agent
from run_experiment import run_experiment
import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
import random
import torch.nn.functional as F
import gym
from PIL import Image
from torchvision import transforms
import copy

# The nn architecture follows the original DQN paper as closely as possible


class DQN(nn.Module):
    def __init__(self, n_actions):
        super(DQN, self).__init__()
        # IMPORTANT!!! these values are hardcoded such that input size must be [x, 4, 84, 84]
        self.conv1 = nn.Conv2d(
            in_channels=4, out_channels=16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=4, stride=2)
        self.fc1 = nn.Linear(32*9*9, 256)
        self.fc2 = nn.Linear(256, n_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, x.size()[1]*x.size()[2]*x.size()[3])
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class dqn_agent(base_agent):
    def __init__(self, actions, experience_replay_size=9000, e_greedy=0.05, learning_rate=0.001, reward_decay=0.9):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.actions = actions
        self.epsilon = e_greedy
        self.gamma = reward_decay
        # frozen param - our target dqn does not learn
        self.dqn_target = DQN(len(actions)).to(device=self.device).double()
        self.dqn_policy = DQN(len(actions)).to(device=self.device).double()
        self.experience_replay = []
        self.experience_replay_capacity = experience_replay_size
        self.optimizer = optim.Adam(
            self.dqn_policy.parameters(), lr=learning_rate)
        self.queue = []

    def get_stacked_tensor(self, s):
        if not self.queue:
            self.queue = [s, s, s, s]
        else:
            self.queue.pop(0)
            self.queue.append(s)
        return torch.stack(self.queue).unsqueeze(dim=0)

    def choose_action(self, s) -> int:
        s = preprocess(s)
        s = self.get_stacked_tensor(s)
        if np.random.uniform() >= self.epsilon:
            # we don't plan to do any backprop during inference, so turn it off to convserve mem
            with torch.no_grad():
                action_values = self.dqn_policy(
                    s.to(device=self.device))
            action = int(torch.argmax(action_values).item())
        else:
            action = np.random.choice(self.actions)
        return action

    def learn(self, s, a, r, s_, done) -> int:
        stacked_s = torch.stack(self.queue)
        a_ = self.choose_action(s_)
        stacked_s_ = torch.stack(self.queue)
        if len(self.experience_replay) >= self.experience_replay_capacity:
            self.experience_replay.pop(0)
        self.experience_replay.append((stacked_s, a, r, stacked_s_))

        if done:
            self.experience_replay[-1] = (stacked_s, a, r, None)
            self.perform_learning_iter()
            self.queue = []
        return a_

    def perform_learning_iter(self):
        random.shuffle(self.experience_replay)
        sample = self.experience_replay[-256:]
        states, actions, rewards, next_states = zip(*sample)

        next_states_non_terminal_mask = torch.tensor(
            [i is not None for i in next_states], dtype=torch.bool).to(device=self.device)
        states = torch.stack(states).double().to(device=self.device)
        # print(states.size())
        actions = torch.tensor(actions).to(device=self.device)
        rewards = torch.tensor(rewards).double().to(device=self.device)
        # print(actions.size())
        non_terminal_next_states = torch.stack(
            [i for i in next_states if i is not None]).double().to(device=self.device)

        # [1,2,3] (size=3) -> unsqueeze(1) -> [[1], [2], [3]] (size=3x1)
        # [[1], [2], [3]] (size=3x1) -> squeeze(1) -> [1,2,3]
        # gather() basically take the value of the actions we have taken
        predicted_values = self.dqn_policy(states).gather(
            1, actions.unsqueeze(1)).squeeze(1)

        next_states_values = torch.zeros(
            rewards.size(), dtype=torch.double).to(device=self.device)
        # make sure that dqn_target's param are frozen
        with torch.no_grad():
            # if we can take two actions, it returns [[#, #],[#, #],...]
            # we take max(1) means take the max value for each [#, #]
            # everything that is not covered by the mask will be defaulted to 0
            next_states_values[next_states_non_terminal_mask] = self.dqn_target(
                non_terminal_next_states).max(1)[0]

        actual_values = rewards + self.gamma*next_states_values

        loss = F.mse_loss(predicted_values, actual_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.dqn_target.load_state_dict(self.dqn_policy.state_dict())

    def save_model(self, path):
        torch.save(self.dqn_policy, path)


def preprocess(s):
    # convert numpy array to PIL image
    im = transforms.ToPILImage()(s)

    # crop frame to remove score
    top_left = (6, 29)  # (x, y)
    bottom_right = (154, 210)
    im = im.crop((top_left[0], top_left[1], bottom_right[0], bottom_right[1]))
    # print(im.size)

    # convert to grayscale
    im_gray = transforms.Grayscale(num_output_channels=1)(im)
    # resize
    im_resized = transforms.Resize(size=(84, 84))(im_gray)
    # normalize to [0,1] and convert PIL image to tensor
    s = transforms.ToTensor()(im_resized).double()
    return s.squeeze()


# def stack_frame(s):
#     return torch.stack([s, s, s, s]).unsqueeze(dim=0)


if __name__ == "__main__":
    # IMPORTANT: the observation space has been hardcoded into the preprocess function and the nn, so it can only be used for breakout env!
    env = gym.make("BreakoutDeterministic-v4")
    env = gym.wrappers.Monitor(env, "Breakout", force=True)
    agent = dqn_agent([i for i in range(env.action_space.n)])
    run_experiment(env, agent, num_eps=int(1e6))
    agent.save_model('dqn-vanilla.pt')

    # env.reset()
    # env.step(env.action_space.sample())
    # env.step(env.action_space.sample())
    # s, _, _, _ = env.step(env.action_space.sample())
    # s = preprocess(s)
    # s_ = stack_frame(s)
    # print(s_.size())
    # print(DQN(env.action_space.n).double()(s_))
    # agent = dqn_agent(env.observation_space.shape[0], [
    #                   i for i in range(env.action_space.n)])
    # run_cartpole_experiment(agent)
    # agent.save_model('dqn-vanilla.pt')
