import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
import random
import torch.nn.functional as F
import gym
from base_agent import base_agent
import matplotlib.pyplot as plt


class DDQN(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super(DDQN, self).__init__()
        self.first_layer = nn.Linear(obs_dim, 128)
        self.second_layer = nn.Linear(128, 64)
        self.output = nn.Linear(64, n_actions)
    
    def forward(self, x):
        x = F.relu(self.first_layer(x))
        x = F.relu(self.second_layer(x))
        return self.output(x)


class Observations:
    def __init__(self, obs_dim, buf_size):
        assert buf_size>1
        self.size = buf_size
        self.obs_dim = obs_dim
        self.state = np.zeros([self.size, self.obs_dim], dtype=np.float32)
        self.reward = np.zeros(self.size, dtype=np.float32)
        self.action = np.zeros(self.size, dtype=np.short)
        self.next_state = np.zeros([self.size, self.obs_dim], dtype=np.float32)
        self.pointer = 0
        self.buf_full = False
    
    def sample(self, size):
        assert size<self.size
        mini_batch_indices = np.random.randint(self.pointer if not self.buf_full else self.size, size=size)
        return (np.take(self.state, mini_batch_indices, axis=0), np.take(self.action, mini_batch_indices), np.take(self.reward, mini_batch_indices), np.take(self.next_state, mini_batch_indices, axis=0))
    
    def append(self, state, action, reward, next_state):
        self.state[self.pointer] = state
        self.action[self.pointer] = action
        self.reward[self.pointer] = reward
        self.next_state[self.pointer] = next_state
        self.pointer = (self.pointer+1)%self.size
        if self.pointer==0:
            self.buf_full=True


class Agent(base_agent):
    def __init__(self, obs_dim, n_actions, er_buf_size=int(1e6), batch_size=32, lr=1e-3, gamma=0.9, epsilon=0.1):
        assert er_buf_size>batch_size
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.obs = Observations(obs_dim, er_buf_size)
        self.n_actions = n_actions
        self.target_network = DDQN(obs_dim, n_actions).to(device=self.device)
        self.policy_network = DDQN(obs_dim, n_actions).to(device=self.device)
        self.epsilon = epsilon
        self.gamma = gamma
        self.batch_size = batch_size
        self.buf_large_enough = False # only start training when buffer size > batch_size
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)

    def choose_action(self, s):
        if np.random.uniform() >= self.epsilon:
            output = self.policy_network(torch.tensor(s, dtype=torch.float32, device=self.device))
            return torch.argmax(output).tolist()
        else: 
            return np.random.randint(self.n_actions)
    
    def learn(self, s, a, r, s_, done):
        self.obs.append(s, a, r, s_ if not done else None)
        # check for none next state: 
        if self.buf_large_enough or self.obs.pointer>self.batch_size:
            self.buf_large_enough = True
            self.update()
        return self.choose_action(s_)
    
    def update(self):
        s, a, r, s_ = self.obs.sample(self.batch_size)
        not_terminal_mask = [not np.isnan(i[0]) for i in s_]

        s = torch.tensor(s, device=self.device)
        a = torch.tensor(a, device=self.device, dtype=torch.int64)
        r = torch.tensor(r, device=self.device)

        q = self.policy_network(s).gather(1, a.unsqueeze(1)).squeeze(1)
        max_q_ = torch.zeros_like(r, device=self.device)
        max_q_[not_terminal_mask], _ = self.target_network(torch.tensor(s_[not_terminal_mask], device=self.device)).max(dim=1)

        td_target = r + self.gamma * max_q_
        loss = F.mse_loss(q, td_target) # poor performance for Huber loss - why?

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.target_network.load_state_dict(self.policy_network.state_dict())

def run_cartpole_experiment(display_plot=True, plot_name=None):
    env = gym.make("CartPole-v0")
    agent = Agent(env.observation_space.shape[0], env.action_space.n)
    r_per_eps = []
    mean_per_100_eps = []
    solved = False
    eps = 0
    while not solved:
        eps += 1
        s = env.reset()  
        a = agent.choose_action(s)
        total_r_in_eps = 0
        while True:
            # take action, observe s' and r
            s_, r, done, _ = env.step(a)
            # get next action
            a_ = agent.learn(s, a, r, s_, done)
            total_r_in_eps += r
            s = s_
            a = a_
            if done:
                r_per_eps.append(total_r_in_eps)
                if len(r_per_eps)%100==0: 
                    mean = np.mean(r_per_eps[-100:])
                    mean_per_100_eps.append(mean)
                    print('Average reward for past 100 episode', mean, 'in episode', eps)
                    if mean>=195:
                        solved = True
                        print('Solved at episode', eps)
                break
    
    r_per_eps_x = [i+1 for i in range(len(r_per_eps))]
    r_per_eps_y = r_per_eps

    mean_per_100_eps_x = [(i+1)*100 for i in range(len(mean_per_100_eps))]
    mean_per_100_eps_y = mean_per_100_eps

    plt.plot(r_per_eps_x, r_per_eps_y, mean_per_100_eps_x, mean_per_100_eps_y)
    if plot_name:
        plt.savefig(plot_name+'.png')
    if display_plot:
        plt.show()
    return r_per_eps


if __name__ == "__main__":
    run_cartpole_experiment()
