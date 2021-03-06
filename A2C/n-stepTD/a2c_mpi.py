from base_agent import base_agent
import torch.nn as nn
import torch
import torch.optim as optim
from torch.distributions import Categorical
import torch.nn.functional as F
import gym
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
from torch.utils.tensorboard import SummaryWriter

name_of_program = 'a2c'
# env = gym.make("LunarLander-v2")
env = gym.make("CartPole-v0")
# env = gym.wrappers.Monitor(env, "recording", force=True)
step_size = 128
num_epoch = 2000
comm = MPI.COMM_WORLD
# get number of processes
num_proc = comm.Get_size()
# get pid
rank = comm.Get_rank()
writer = SummaryWriter(f'runs/Proc {rank}')

# A3C paper: https://arxiv.org/pdf/1602.01783.pdf
class ActorCritic(nn.Module):
    def __init__(self, input_size, n_actions):
        super(ActorCritic, self).__init__()
        self.first_layer = nn.Linear(input_size, 128)
        self.critic_layer = nn.Linear(128, 64)
        self.actor_layer = nn.Linear(128, 32)
        self.critic_head = nn.Linear(64, 1)
        self.actor_head = nn.Linear(32, n_actions)

    def forward(self, x):
        x = F.relu(self.first_layer(x))
        actor_out = F.relu(self.actor_layer(x))
        critic_out = F.relu(self.critic_layer(x.detach()))
        return self.actor_head(actor_out), self.critic_head(critic_out)

class Observations():
    def __init__(self, step_size, obs_dim):
        self.size = step_size
        self.obs_dim = obs_dim
        self.reset()

    def reset(self):
        self.state = np.zeros([self.size, self.obs_dim])
        self.reward = np.zeros(self.size)
        self.action = np.zeros(self.size)
        self.counter, self.start_of_eps = 0, 0

    def append(self, state, action, reward):
        assert self.counter<self.size
        self.state[self.counter] = state
        self.reward[self.counter] = reward
        self.action[self.counter] = action
        self.counter += 1

    def compute_discounted_rewards(self, gamma, next_state_val):
        cumulative_reward = np.zeros_like(self.reward[self.start_of_eps:])
        if len(cumulative_reward)<2:
            self.start_of_eps = self.counter
            return
        cumulative_reward[0] = self.reward[self.counter-1]+gamma*next_state_val
        j = 1
        for i in reversed(range(self.start_of_eps, self.counter-1)):
            cumulative_reward[j] = self.reward[i] + gamma*cumulative_reward[j-1]
            j += 1
        self.reward[self.start_of_eps:] = cumulative_reward[::-1]
        self.start_of_eps = self.counter


class Agent(base_agent):
    def __init__(self, env, n_step=10, reward_decay=0.9, lr=5e-4):
        # self.device = torch.device(
        #     'cuda' if torch.cuda.is_available() else 'cpu')
        global comm
        self.device = torch.device('cpu')
        self.gamma = reward_decay
        self.agent = ActorCritic(
            env.observation_space.shape[0], env.action_space.n).double().to(device=self.device)
        state_dict = comm.bcast(self.agent.state_dict(), root=0)
        self.agent.load_state_dict(state_dict)
        self.optim = optim.Adam(self.agent.parameters(), lr=lr)
        self.obs = Observations(n_step, env.observation_space.shape[0])
        self.n_step = n_step

    def choose_action(self, s) -> int:
        s = torch.tensor(s, dtype=torch.double).to(device=self.device)
        actor_logits, _ = self.agent(s)
        probs = Categorical(logits=actor_logits)
        a = probs.sample()
        return a.tolist()

    def learn(self, s, a, r, s_, done) -> int:
        self.obs.append(s, a, r)
        if done:
            self.obs.compute_discounted_rewards(self.gamma, 0)
        return 0 if done else self.choose_action(s_)

    def perform_learning_iter(self, s_, done):
        val = 0
        if not done:
            _, val = self.agent(torch.tensor(s_, dtype=torch.double).to(device=self.device))
        self.obs.compute_discounted_rewards(self.gamma, val)
        assert self.obs.counter==self.n_step 
        states = torch.tensor(self.obs.state, dtype=torch.double).to(
            device=self.device)
        actions = torch.tensor(self.obs.action).to(device=self.device)
        discounted_rewards = torch.tensor(
            self.obs.reward, device=self.device).squeeze()
        actor_logits, critic_values = self.agent(states)
        probs = Categorical(logits=actor_logits)
        log_probs = probs.log_prob(actions)
        entropy = probs.entropy().mean()

        critic_values = critic_values.squeeze()
        
        delta = discounted_rewards - critic_values
        mean, std = self.mpi_statistics_scalar(delta.detach().numpy())
        delta = (delta-mean)/std

        # actor_loss = (-log_probs*(delta.detach())-0.01*entropy).mean()
        actor_loss = (-log_probs*(delta.detach())).mean()
        critic_loss = F.smooth_l1_loss(critic_values.float(), discounted_rewards.float())
        loss = actor_loss + 0.1*critic_loss
        self.optim.zero_grad()
        loss.backward()

        for p in self.agent.parameters():
            p_grad_numpy = p.grad.numpy()
            avg_p_grad = self.mpi_avg(p.grad)
            p_grad_numpy[:] = avg_p_grad[:]

        self.optim.step()
        self.obs.reset()
        return actor_loss, critic_loss, loss, entropy

    # Implementation of MPI functions are exact copies of that of Spinning Up's
    def mpi_avg(self, x):
        global num_proc
        return self.mpi_sum(x) / num_proc

    def mpi_sum(self, x):
        global comm
        x, scalar = ([x], True) if np.isscalar(x) else (x, False)
        x = np.asarray(x, dtype=np.float32)
        buff = np.zeros_like(x, dtype=np.float32)
        # Take sum and distribute them to all processes
        comm.Allreduce(x, buff, op=MPI.SUM)
        return buff[0] if scalar else buff

    def mpi_statistics_scalar(self, x):
        x = np.array(x, dtype=np.float32)
        global_sum, global_n = self.mpi_sum([np.sum(x), len(x)])
        mean = global_sum / global_n

        var = (self.mpi_sum(np.sum((x - mean)**2)))/global_n
        std = np.sqrt(var)

        return mean, std
    
    def getAgent(self):
        return self.agent


def run(env, num_epoch, n_step):
    agent = Agent(env, n_step=n_step)
    r_per_eps = []
    s = env.reset()
    a = agent.choose_action(s)
    total_r_in_eps = 0
    done = False
    for epoch in range(num_epoch):
        for _ in range(n_step):
            s_, r, done, _ = env.step(a)
            a_ = agent.learn(s, a, r, s_, done)
            total_r_in_eps += r
            s = s_
            a = a_
            if done:
                r_per_eps.append(total_r_in_eps)
                writer.add_scalars(f'process_{rank}/Reward', {f"reward_per_eps_{name_of_program}":total_r_in_eps}, len(r_per_eps)-1)
                if len(r_per_eps) % 100 == 0:
                    mean = np.mean(r_per_eps[-100:])
                    var = np.var(r_per_eps[-100:])
                    print(f'{rank}: Average reward for past 100 episode {mean} in episode {len(r_per_eps)}', flush=True)
                    writer.add_scalars(f'process_{rank}/Reward', {f"mean_{name_of_program}":mean, f"var_{name_of_program}":var, f"reward_per_eps_{name_of_program}":total_r_in_eps}, len(r_per_eps)-1)
                s = env.reset()
                a = agent.choose_action(s)
                total_r_in_eps = 0
        actor_loss, critic_loss, loss, entropy = agent.perform_learning_iter(s_, done)
        writer.add_scalars(f'process_{rank}/Actor Loss', {f"{name_of_program}":actor_loss}, epoch)
        writer.add_scalars(f'process_{rank}/Crtic Loss', {f"{name_of_program}":critic_loss}, epoch)
        writer.add_scalars(f'process_{rank}/Combined Loss', {f"{name_of_program}":loss}, epoch)
        writer.add_scalars(f'process_{rank}/Entropy', {f"{name_of_program}":entropy}, epoch)
    return r_per_eps, agent.getAgent()


if __name__ == "__main__":

    r_per_eps, agent = run(env, num_epoch, step_size)
    # reward = comm.gather(r_per_eps, root=0)
    writer.close()
    if rank==0:
        torch.save(agent, 'nstep-A2C.pt')
        # reward = r_per_eps
        # # reward = [i for i in reward][0]
        # # print(reward)
        # r_per_eps_x = [i+1 for i in range(len(reward))]
        # r_per_eps_y = reward
        # mean_per_100_eps = [np.mean(reward[i:i+100])
        #                     for i in range(0, len(reward)-100, 100)]
        # mean_per_100_eps_x = [(i+1)*100 for i in range(len(mean_per_100_eps))]
        # mean_per_100_eps_y = mean_per_100_eps

        # plt.plot(r_per_eps_x, r_per_eps_y, mean_per_100_eps_x, mean_per_100_eps_y)
        # plt.show()
