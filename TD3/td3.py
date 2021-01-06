from copy import deepcopy
import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
import random
import torch.nn.functional as F
import gym
import matplotlib.pyplot as plt
import itertools


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


class ActorCritic(nn.Module): 
    # we inherit nn.Module so that we can get all param using actorcritic.parameters()
    def __init__(self, obs_dim, act_dim, act_limit):
        super().__init__()
        self.pi = Actor(obs_dim, act_dim, act_limit)
        self.q1 = QFunction(obs_dim, act_dim)
        self.q2 = QFunction(obs_dim, act_dim)


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


class Agent:
    def __init__(self, obs_dim, act_dim, act_limit, er_buf_size=int(1e6), batch_size=100, q_lr=1e-3, pi_lr=1e-3, gamma=0.9, act_noise=0.1, noise_clip=0.5, target_noise=0.2, polyak=0.995, start_steps=10000, policy_delay=2):
        assert er_buf_size>batch_size
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.obs = Observations(obs_dim, act_dim, er_buf_size)
        self.act_dim = act_dim
        self.act_limit = act_limit
        self.ac = ActorCritic(obs_dim, act_dim, act_limit).to(device=self.device)
        self.ac_target = deepcopy(self.ac)
        self.act_noise = act_noise # noise when taking actions
        self.target_noise = target_noise # noise when bootstrapping
        self.noise_clip = noise_clip # max noise for td target actions
        self.polyak = polyak
        self.gamma = gamma
        self.policy_delay = policy_delay
        self.batch_size = batch_size # performance collapse if batch size is too large
        self.buf_large_enough = False # only start training when buffer size > batch_size
        self.q_params = itertools.chain(self.ac.q1.parameters(), self.ac.q2.parameters())
        self.pi_optimizer = optim.Adam(self.ac.pi.parameters(), lr=pi_lr)
        self.q_optimizer = optim.Adam(self.q_params, lr=q_lr)
        self.timer = 0

    def choose_action(self, s):
        a = self.ac.pi(torch.tensor(s, dtype=torch.float32, device=self.device)).cpu()
        a += self.act_noise*torch.rand(self.act_dim)
        return torch.clip(a, -self.act_limit, self.act_limit).tolist()
    
    def test_phase_choose_action(self, s):
        a = self.ac.pi(torch.tensor(s, dtype=torch.float32, device=self.device)).cpu()
        return torch.clip(a, -self.act_limit, self.act_limit).tolist()

    def learn(self, s, a, r, s_, done):
        self.obs.append(s, a, r, s_ if not done else None)
        # check for none next state: 
        if (self.buf_large_enough or self.obs.pointer>self.batch_size): # performance collapse if update on every step
            self.timer += 1
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

        def update_q_funct():
            # Compute target actions
            target_a = self.ac_target.pi(non_terminal_s_)
            epsilon = torch.randn_like(target_a)*self.target_noise
            epsilon = torch.clip(epsilon, -self.noise_clip, self.noise_clip)
            target_a = torch.clip(target_a+epsilon, -self.act_limit, self.act_limit)

            # Compute TD target
            td_target = torch.zeros_like(r, device=self.device)
            q1_targ = torch.zeros_like(r, device=self.device)
            q2_targ = torch.zeros_like(r, device=self.device)
            q1_targ[not_terminal_mask] = self.ac_target.q1(non_terminal_s_, target_a)
            q2_targ[not_terminal_mask] = self.ac_target.q2(non_terminal_s_, target_a)
            td_target = r + self.gamma * torch.min(q1_targ, q2_targ).detach()

            q1 = self.ac.q1(s, a)
            q2 = self.ac.q2(s, a)
            loss_q = F.mse_loss(q1, td_target) + F.mse_loss(q2, td_target) 

            # Optimize val funct
            self.q_optimizer.zero_grad()
            loss_q.backward()
            self.q_optimizer.step()
        
        def update_policy():
            # Compute loss for policy (actor)
            q_policy = self.ac.q1(s, self.ac.pi(s))
            loss_pi = -q_policy.mean()

            # Freeze q1 network since we only want to optimize policy here
            for p in self.q_params:
                p.requires_grad = False
            
            # Optimize policy
            self.pi_optimizer.zero_grad()
            loss_pi.backward()
            self.pi_optimizer.step()

            # Unfreeze q1 network after we are done optimizing policy 
            for p in self.q_params:
                p.requires_grad = True

        update_q_funct()
        if self.timer%self.policy_delay==0:
            self.timer = 0
            update_policy()
            # update target network
            for act, targ in zip(self.ac.parameters(), self.ac_target.parameters()):
                targ.data = (targ.data*self.polyak) + (act.data*(1-self.polyak))
            



def run_cartpole_experiment(display_plot=True, plot_name=None):
    # env = gym.make("LunarLanderContinuous-v2")
    env = gym.make("Pendulum-v0")
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
