from copy import deepcopy
import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
import random
import torch.nn.functional as F
import gym
from base_agent import base_agent
import matplotlib.pyplot as plt


class QFunction(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.input = nn.Linear(obs_dim+act_dim, 128)
        self.fc1 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 1)

    def forward(self, obs, act):
        x = F.relu(self.input(torch.cat([obs, act], dim=-1)))
        x = F.relu(self.fc1(x))
        return self.output(x).squeeze(-1)


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, act_limit):
        super().__init__()
        self.input = nn.Linear(obs_dim, 128)
        self.fc1 = nn.Linear(128, 64)
        self.output = nn.Linear(64, act_dim)
        self.act_limit = act_limit

    def forward(self, obs):
        x = F.relu(self.input(obs))
        x = F.relu(self.fc1(x))
        return self.act_limit * torch.tanh(self.output(x)) # scale output according to action space

class Observations:
    def __init__(self, obs_dim, act_dim, buf_size):
        assert buf_size>1
        self.max_size = buf_size
        self.obs_dim = obs_dim
        self.state = np.zeros([self.max_size, self.obs_dim], dtype=np.float32)
        self.reward = np.zeros(self.max_size, dtype=np.float32)
        self.action = np.zeros([self.max_size, act_dim], dtype=np.float32)
        self.next_state = np.zeros([self.max_size, self.obs_dim], dtype=np.float32)
        self.pointer = 0
        self.curr_size = 0
    
    def sample(self, size):
        assert size<=self.curr_size
        mini_batch_indices = np.random.randint(self.curr_size, size=size)
        return (
            np.take(self.state, mini_batch_indices, axis=0), 
            np.take(self.action, mini_batch_indices, axis=0), 
            np.take(self.reward, mini_batch_indices), 
            np.take(self.next_state, mini_batch_indices, axis=0)
        )
    
    def append(self, state, action, reward, next_state):
        self.state[self.pointer] = state
        self.action[self.pointer] = action
        self.reward[self.pointer] = reward
        self.next_state[self.pointer] = next_state
        self.pointer = (self.pointer+1)%self.max_size
        self.curr_size = min(self.curr_size+1, self.max_size)


class Agent(base_agent):
    def __init__(self, obs_dim, act_dim, act_limit, er_buf_size=int(1e6), batch_size=100, q_lr=1e-3, pi_lr=1e-3, gamma=0.9, act_noise=0.1, polyak=0.995, start_steps=10000):
        assert er_buf_size>batch_size
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.obs = Observations(obs_dim, act_dim, er_buf_size) # if buf size is too large performance drops - likely because we will be reusing too much old, suboptimal data
        self.act_dim = act_dim
        self.act_limit = act_limit
        self.q = QFunction(obs_dim, act_dim).to(device=self.device)
        self.act = Actor(obs_dim, act_dim, act_limit).to(device=self.device)
        self.q_target = deepcopy(self.q)
        self.act_target = deepcopy(self.act)
        self.act_noise = act_noise
        self.polyak = polyak
        self.gamma = gamma
        self.batch_size = batch_size # performance collapse if batch size is too large
        self.buf_large_enough = False # only start training when buffer size > batch_size
        self.pi_optimizer = optim.Adam(self.act.parameters(), lr=pi_lr)
        self.q_optimizer = optim.Adam(self.q.parameters(), lr=q_lr)

    def choose_action(self, s, noise_scale=0.1):
        a = self.act(torch.tensor(s, dtype=torch.float32, device=self.device)).cpu()
        a += noise_scale*torch.rand(self.act_dim)
        return torch.clip(a, -self.act_limit, self.act_limit).tolist()
    
    def test_phase_choose_action(self, s):
        return self.choose_action(s, noise_scale=0)

    def learn(self, s, a, r, s_, done):
        self.obs.append(s, a, r, s_ if not done else None)
        # check for none next state: 
        if (self.buf_large_enough or self.obs.pointer>self.batch_size): # performance collapse if update on every step
            self.buf_large_enough = True
            self.update()
        return self.choose_action(s_)
    
    def update(self):
        s, a, r, s_ = self.obs.sample(self.batch_size)
        not_terminal_mask = [not np.isnan(i[0]) for i in s_]

        s = torch.tensor(s, device=self.device)
        a = torch.tensor(a, device=self.device)
        r = torch.tensor(r, device=self.device)
        non_terminal_s_ = torch.tensor(s_[not_terminal_mask], device=self.device)

        # Compute loss for val funct (q)
        q_val = self.q(s, a)
        max_q_ = torch.zeros_like(r, device=self.device)
        max_q_[not_terminal_mask] = self.q_target(non_terminal_s_, self.act_target(non_terminal_s_)).detach()
        td_target = r + self.gamma * max_q_
        loss_q = F.mse_loss(q_val, td_target) 

        # Optimize val funct
        self.q_optimizer.zero_grad()
        loss_q.backward()
        self.q_optimizer.step()
        
        # Compute loss for policy (actor)
        q_policy = self.q(s, self.act(s))
        loss_pi = -q_policy.mean()

        # Freeze q network since we only want to optimize policy here
        for p in self.q.parameters():
            p.requires_grad = False
        
        # Optimize policy
        self.pi_optimizer.zero_grad()
        loss_pi.backward()
        self.pi_optimizer.step()

        # Unfreeze q network after we are done optimizing policy 
        for p in self.q.parameters():
            p.requires_grad = True

        def update_network_polyak(actual, target):
            for act, targ in zip(actual.parameters(), target.parameters()):
                targ.data = (targ.data*self.polyak) + (act.data*(1-self.polyak))

        # Update target networks using polyak averaging
        with torch.no_grad():
            update_network_polyak(self.q, self.q_target)
            update_network_polyak(self.act, self.act_target)
            



def run_cartpole_experiment(display_plot=True, plot_name=None):
    # env = gym.make("MountainCarContinuous-v0")
    env = gym.make("BipedalWalker-v3")
    agent = Agent(env.observation_space.shape[0], env.action_space.shape[0], env.action_space.high[0])
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
                    if mean>=-180:
                        print(f'Solved* Pendulum in episode {eps}')
                        solved = True
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
