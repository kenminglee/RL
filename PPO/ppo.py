from base_agent import base_agent
import torch.nn as nn
import torch
import torch.optim as optim
import torch.multiprocessing as mp
from torch.distributions import Categorical
from torch.multiprocessing import Queue, Lock, Value, Array
import torch.nn.functional as F
import gym
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI

# env = gym.make("LunarLander-v2")
env = gym.make("CartPole-v0")

step_size = 128

comm = MPI.COMM_WORLD
# get number of processes
num_proc = comm.Get_size()
# get pid
rank = comm.Get_rank()

# A3C paper: https://arxiv.org/pdf/1602.01783.pdf
class ActorCritic(nn.Module):
    def __init__(self, input_size, n_actions):
        super(ActorCritic, self).__init__()
        self.first_layer = nn.Linear(input_size, 128)
        self.critic_head = nn.Linear(128, 1)
        self.actor_head = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.first_layer(x))
        return self.actor_head(x), self.critic_head(x.detach())

class Observations():
    def __init__(self):
        if rank==0:
            self.size = step_size*num_proc
        else:
            self.size = step_size
        self.reset()

    def reset(self):
        self.state = np.zeros(size)
        self.logp = np.zeros(size)
        self.reward = np.zeros(size)
        self.entropy = np.zeros(size)
        self.counter = 0

    def append(self, state, logp, reward, entropy):
        assert self.counter<self.size
        self.state[self.counter] = state
        self.logp[self.counter] = logp
        self.reward[self.counter] = reward
        self.entropy[self.counter] = entropy
        self.counter += 1

class Agent(base_agent):
    def __init__(self, env, n_step=10, reward_decay=0.9):
        # self.device = torch.device(
        #     'cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cpu')
        self.gamma = reward_decay
        self.agent = ActorCritic(
            env.observation_space.shape[0], env.action_space.n).double().to(device=self.device)
        state_dict = comm.bcast(agent.state_dict(), root=0)
        self.agent.load_state_dict(state_dict)
        self.optim = optim.Adam(self.agent.parameters())
        self.obs = Observations()
        self.n_step = n_step
        self.latest_log_prob = []
        self.latest_entropy = []

    def choose_action(self, s) -> int:
        s = torch.tensor(s, dtype=torch.double).to(device=self.device)
        actor_logits, _ = self.agent(s)
        probs = Categorical(logits=actor_logits)
        a = probs.sample()
        assert len(self.latest_log_prob) == 0 and len(self.latest_entropy)==0
        self.latest_log_prob.append(probs.log_prob(a))
        self.latest_entropy.append(probs.entropy())
        return a.tolist()

    def learn(self, s, a, r, s_, done) -> int:
        self.obs.append(s, self.latest_log_prob.pop(), r, self.latest_entropy.pop())
        if done or len(self.obs) >= self.n_step:
            self.perform_learning_iter(s_, done)
            self.obs = []
        return 0 if done else self.choose_action(s_)

    def perform_learning_iter(self, s_, done):
        self.optim.zero_grad()
        states, log_probs, rewards, entropy = zip(*self.obs)
        states = torch.tensor(states, dtype=torch.double).to(
            device=self.device)
        if not done:
            _, val = self.agent(torch.tensor(
                s_, dtype=torch.double).to(device=self.device))
            discounted_rewards = self.convert_to_discounted_reward(
                rewards, val.squeeze())
        else:
            discounted_rewards = self.convert_to_discounted_reward(rewards, 0)
        discounted_rewards = torch.tensor(
            discounted_rewards, device=self.device).squeeze()
        log_probs = torch.stack(
            log_probs).to(device=self.device).squeeze()
        entropy = torch.stack(entropy).to(device=self.device).squeeze()
        _, critic_values = self.agent(states)
        critic_values = critic_values.squeeze()
        
        delta = discounted_rewards - critic_values
        loss = (torch.sum(-log_probs*(delta.detach())-0.01*entropy) \
            + 0.1*F.smooth_l1_loss(critic_values.float(), discounted_rewards.float()))/self.process_num

        self.agent.load_state_dict(self.gl_agent.state_dict())

    def convert_to_discounted_reward(self, rewards, next_state_val):
        cumulative_reward = [rewards[-1]+self.gamma*next_state_val]
        for reward in reversed(rewards[:-1]):
            cumulative_reward.append(reward + self.gamma*cumulative_reward[-1])
        return cumulative_reward[::-1]


def run(env, n_step):
    
    # env = gym.wrappers.Monitor(env, "recording", force=True)
    agent = Agent(env, gl_agent, gl_lock, semaphore, gl_count, num_processes, n_step=n_step)
    r_per_eps = []
    while not bool(gl_solved.value):
        s = env.reset()
        a = agent.choose_action(s)
        total_r_in_eps = 0
        while not bool(gl_solved.value):
            s_, r, done, _ = env.step(a)
            a_ = agent.learn(s, a, r, s_, done)
            total_r_in_eps += r
            s = s_
            a = a_
            if done:
                r_per_eps.append(total_r_in_eps)
                gl_r_per_eps.put(total_r_in_eps)
                if len(r_per_eps) % 100 == 0:
                    mean = np.mean(r_per_eps[-100:])
                    gl_print_lock.acquire()
                    print(worker_num, ': Average reward for past 100 episode',
                          mean, 'in episode', len(r_per_eps))
                    gl_print_lock.release()
                    if mean >= 195:
                        with gl_solved.get_lock():
                            gl_solved.value = True
                        gl_print_lock.acquire()
                        print(worker_num, ': Solved at episode', len(r_per_eps))
                        gl_print_lock.release()
                break


if __name__ == "__main__":

    

    run(env)

    r_per_eps_x = [i+1 for i in range(len(reward))]
    r_per_eps_y = reward
    mean_per_100_eps = [np.mean(reward[i:i+100])
                        for i in range(0, len(reward)-100, 100)]
    mean_per_100_eps_x = [(i+1)*100 for i in range(len(mean_per_100_eps))]
    mean_per_100_eps_y = mean_per_100_eps

    plt.plot(r_per_eps_x, r_per_eps_y, mean_per_100_eps_x, mean_per_100_eps_y)
    plt.show()
