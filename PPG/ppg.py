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

name_of_program = "ppg"
# env = gym.make("LunarLander-v2")
env = gym.make("CartPole-v0")

step_size = 128
num_epoch = 2000
comm = MPI.COMM_WORLD
# get number of processes
num_proc = comm.Get_size()
# get pid
rank = comm.Get_rank()
writer = SummaryWriter(f'runs/Proc {rank}')
# if rank==0:
#     env = gym.wrappers.Monitor(env, "recording", force=True)

class Actor(nn.Module):
    def __init__(self, input_size, n_actions):
        super(Actor, self).__init__()
        self.first_layer = nn.Linear(input_size, 128)
        self.second_layer = nn.Linear(128, 64)
        self.actor_head = nn.Linear(64, n_actions)
        self.critic_head = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.first_layer(x))
        x = F.relu(self.second_layer(x))
        return self.actor_head(x), self.critic_head(x)

class Critic(nn.Module):
    def __init__(self, input_size, n_actions):
        super(Critic, self).__init__()
        self.first_layer = nn.Linear(input_size, 128)
        self.second_layer = nn.Linear(128, 64)
        self.critic_head = nn.Linear(64, 1)
    
    def forward(self, x):
        x = F.relu(self.first_layer(x))
        x = F.relu(self.second_layer(x))
        return self.critic_head(x)

class Observations():
    def __init__(self, step_size, obs_dim):
        self.size = step_size
        self.obs_dim = obs_dim
        self.state = np.zeros([self.size, self.obs_dim], dtype=np.float32)
        self.reward = np.zeros(self.size, dtype=np.float32)
        self.action = np.zeros(self.size)
        self.reset()

    def reset(self):
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
    def __init__(self, env, n_step=10, reward_decay=0.99, clip_ratio=0.2, 
    policy_epoch=1, value_epoch=1, aux_epoch=6, lr=5e-4, beta_clone=1):
        # self.device = torch.device(
        #     'cuda' if torch.cuda.is_available() else 'cpu')
        global comm
        self.device = torch.device('cpu')
        self.epsilon = clip_ratio
        self.gamma = reward_decay
        self.policy_epoch = policy_epoch
        self.value_epoch = value_epoch
        self.aux_epoch = aux_epoch
        self.beta_clone = beta_clone
        self.actor = Actor(
            env.observation_space.shape[0], env.action_space.n).to(device=self.device)
        self.critic = Critic(
            env.observation_space.shape[0], env.action_space.n).to(device=self.device)
        actor_state_dict = comm.bcast(self.actor.state_dict(), root=0)
        self.actor.load_state_dict(actor_state_dict)
        critic_state_dict = comm.bcast(self.critic.state_dict(), root=0)
        self.critic.load_state_dict(critic_state_dict)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=lr)
        self.obs = Observations(n_step, env.observation_space.shape[0])
        self.n_step = n_step

    def choose_action(self, s) -> int:
        s = torch.tensor(s, dtype=torch.float32).to(device=self.device)
        actor_logits, _ = self.actor(s)
        probs = Categorical(logits=actor_logits)
        a = probs.sample()
        return a.tolist()

    def learn(self, s, a, r, s_, done) -> int:
        self.obs.append(s, a, r)
        if done:
            self.obs.compute_discounted_rewards(self.gamma, 0)
        return 0 if done else self.choose_action(s_)

    def update(self, s_, done, epoch, update_aux=False):
        global rank, name_of_program
        val = 0
        if not done:
            val = self.critic(torch.tensor(s_, dtype=torch.float32).to(device=self.device))
        self.obs.compute_discounted_rewards(self.gamma, val)
        assert self.obs.counter==self.n_step 
        states = torch.tensor(self.obs.state).float().to(
            device=self.device)
        actions = torch.tensor(self.obs.action).to(device=self.device)
        mean, std = self.mpi_statistics_scalar(self.obs.reward)
        discounted_rewards = torch.tensor(
            self.obs.reward, device=self.device).squeeze()
        discounted_rewards = ((discounted_rewards-mean)/std)
        
        self.update_policy_phase(states, actions, discounted_rewards, epoch)
        if update_aux:
            self.update_aux_phase(states, actions, discounted_rewards, epoch)
        self.obs.reset()
        
    
    def update_policy_phase(self, states, actions, discounted_rewards, epoch):
        actor_logits, _ = self.actor(states)
        critic_values = self.critic(states)
        old_log_probs = Categorical(logits=actor_logits).log_prob(actions).detach()
        
        # Calc advantage
        delta = discounted_rewards - critic_values.squeeze()
        mean, std = self.mpi_statistics_scalar(delta.detach().numpy())
        delta = ((delta-mean)/std).detach()

        def calc_policy_loss():
            actor_logits, _ = self.actor(states)
            dist = Categorical(logits=actor_logits)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean() # Makes it worse for cartpole env.

            # note that torch.exp(log_probs-log_probs_old) == probs/probs_old 
            ratio = torch.exp(log_probs - old_log_probs)
            clip_adv = torch.clamp(ratio, 1-self.epsilon, 1+self.epsilon) * delta
            actor_loss = -(torch.min(ratio*delta, clip_adv)).mean()
            
            approx_kl = (log_probs - old_log_probs).mean().item()
            return actor_loss, approx_kl, entropy
        

        for _ in range(self.policy_epoch): 
            actor_loss, approx_kl, entropy = calc_policy_loss()
            kl = self.mpi_avg(approx_kl)
            if kl > 1.5*0.01:
                print('Early Stopping', flush=True)
                break
            
            self.actor_optim.zero_grad() 
            actor_loss.backward()
            for p in self.actor.parameters():
                if p.grad is not None: # during policy phase aux head will not be reached!
                    p_grad_numpy = p.grad.numpy()
                    avg_p_grad = self.mpi_avg(p.grad)
                    p_grad_numpy[:] = avg_p_grad[:]
            self.actor_optim.step()
        
        for _ in range(self.value_epoch):
            self.update_critic_once(states, actions, discounted_rewards, epoch)

        # writer.add_scalars(f'process_{rank}/Actor Loss', {f"{name_of_program}":tot_actor_loss}, epoch)
        # writer.add_scalars(f'process_{rank}/Crtic Loss', {f"{name_of_program}":tot_critic_loss}, epoch)
        # writer.add_scalars(f'process_{rank}/Combined Loss', {f"{name_of_program}":tot_loss}, epoch)
        # writer.add_scalars(f'process_{rank}/Entropy', {f"{name_of_program}":tot_entropy}, epoch)

    def update_aux_phase(self, states, actions, discounted_rewards, epoch):
        old_actor_logits, _ = self.actor(states)
        old_probs = Categorical(logits=old_actor_logits).probs.detach()
        for _ in range(self.aux_epoch):
            actor_logits, aux = self.actor(states)
            aux_loss = F.smooth_l1_loss(aux.squeeze(), discounted_rewards)
            log_probs = Categorical(logits=actor_logits).probs.log()
            kl = self.beta_clone * F.kl_div(log_probs, old_probs)
            joint_loss = 0.1*(aux_loss + kl)

            self.actor_optim.zero_grad() 
            joint_loss.backward()
            for p in self.actor.parameters():
                p_grad_numpy = p.grad.numpy()
                avg_p_grad = self.mpi_avg(p.grad)
                p_grad_numpy[:] = avg_p_grad[:]
            self.actor_optim.step()
            
            self.update_critic_once(states, actions, discounted_rewards, epoch)

    def update_critic_once(self, states, actions, discounted_rewards, epoch):
        critic_values = self.critic(states)
        critic_loss = F.smooth_l1_loss(critic_values.squeeze(), discounted_rewards)

        self.critic_optim.zero_grad()  
        critic_loss.backward()
        for p in self.critic.parameters():
            p_grad_numpy = p.grad.numpy()
            avg_p_grad = self.mpi_avg(p.grad)
            p_grad_numpy[:] = avg_p_grad[:]
        self.critic_optim.step()

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
        return self.actor


def run(env, num_epoch, n_step, aux_update_freq=32):
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
        agent.update(s_, done, epoch, update_aux=epoch%aux_update_freq==0)

        
    return r_per_eps, agent.getAgent()


if __name__ == "__main__":

    r_per_eps, agent = run(env, num_epoch, step_size)
    # reward = comm.gather(r_per_eps, root=0)
    writer.close()
    if rank==0:
        torch.save(agent, 'ppg.pt')
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