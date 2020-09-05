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
import os
from os import path


class DQN(nn.Module):
    # The nn architecture follows the original DQN paper as closely as possible
    # The major difference is the addition of Dueling architecture https://arxiv.org/pdf/1511.06581.pdf
    def __init__(self, n_actions):
        super(DQN, self).__init__()
        # IMPORTANT!!! these values are hardcoded such that input size must be [x, 4, 84, 84]
        self.n_actions = n_actions
        self.conv1 = nn.Conv2d(
            in_channels=4, out_channels=16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=4, stride=2)
        self.fc1 = nn.Linear(32*9*9, 256)
        self.fc_state_action_val_output = nn.Linear(256, n_actions)
        self.fc_state_val_output = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, x.size()[1]*x.size()[2]*x.size()[3])
        x = F.relu(self.fc1(x))
        stream1_out = self.fc_state_action_val_output(x)
        stream2_out = self.fc_state_val_output(x)
        stream1_out_mean = torch.sum(
            stream1_out, dim=1, keepdim=True) / self.n_actions
        output = stream2_out + (stream1_out - stream1_out_mean)
        return output


class dqn_agent(base_agent):
    def __init__(self, actions, experience_replay_size=10000, e_greedy=0.05, learning_rate=0.001, reward_decay=0.9):
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
            1, actions.unsqueeze(1)).squeeze(1).to(device=self.device)

        next_states_values = torch.zeros(
            rewards.size(), dtype=torch.double).to(device=self.device)
        # make sure that dqn_target's param are frozen
        with torch.no_grad():
            # Double DQN
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

    def load_model(self, path):
        self.dqn_policy = torch.load(path)
        self.dqn_policy.eval()

    def save_checkpoint(self, episode, r_per_eps, max_chkpoints=3, folder_name='checkpoint'):
        MODEL_NAME = 'model_'+str(episode)+'.tar'
        if not path.exists(folder_name):
            os.makedirs(folder_name)
        os.chdir(folder_name)
        existingCheckpoints = [i for i in os.listdir() if 'model_' in i]
        existingCheckpoints = sorted(existingCheckpoints, key=lambda x: int(
            x[x.find('_')+1: x.find('.tar')]), reverse=True)
        filesToDel = existingCheckpoints[max_chkpoints:]
        for fileToDel in filesToDel:
            os.remove(fileToDel)
        torch.save({
            'eps': episode,
            'model_state_dict': self.dqn_policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'experience_replay_buffer': self.experience_replay,
            'r_per_eps': r_per_eps
        }, MODEL_NAME)
        os.chdir('..')

    def load_latest_checkpoint(self, folder_name='checkpoint'):
        os.chdir(folder_name)
        existingCheckpoints = [i for i in os.listdir() if 'model_' in i]
        existingCheckpoints = sorted(existingCheckpoints, key=lambda x: int(
            x[x.find('_')+1: x.find('.tar')]), reverse=True)
        os.chdir('..')
        return self.load_checkpoint(existingCheckpoints[0])

    def load_checkpoint(self, model_path, folder_name='checkpoint'):
        checkpoint = torch.load(folder_name+'/'+model_path)
        self.dqn_policy.load_state_dict(checkpoint['model_state_dict'])
        self.dqn_target.load_state_dict(checkpoint['model_state_dict'])
        self.dqn_policy.train()
        self.dqn_policy.to(device=self.device)
        self.dqn_target.to(device=self.device)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.experience_replay = checkpoint['experience_replay_buffer']
        print('loaded model successfully from episode', checkpoint['eps'])
        return self, checkpoint['eps'], checkpoint['r_per_eps']


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

    # If starting from scratch
    # RECORDING_FOLDER_NAME = "Breakout-eps0"
    # env = gym.wrappers.Monitor(
    #     env, RECORDING_FOLDER_NAME, video_callable=lambda x: x % 1000 == 0)
    # agent = dqn_agent([i for i in range(env.action_space.n)],
    #                   experience_replay_size=9000)
    # run_experiment(env, agent, num_eps=int(
    #     1e6), save_checkpoint_exist=True, save_every_x_eps=1000)

    # If restoring from checkpoint
    RECORDING_FOLDER_NAME = "Breakout-eps15000"
    env = gym.wrappers.Monitor(
        env, RECORDING_FOLDER_NAME, video_callable=lambda x: x % 1000 == 0)
    agent, start_eps, r_per_eps = dqn_agent([i for i in range(env.action_space.n)],
                                            experience_replay_size=10000).load_latest_checkpoint()
    run_experiment(env, agent, num_eps=int(
        1e6), save_checkpoint_exist=True, save_every_x_eps=1000, r_per_eps=r_per_eps, initial_eps=start_eps)

    # agent.save_model('ddqn.pt')
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
