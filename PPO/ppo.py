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


class Agent(base_agent):
    def __init__(self, env, gl_agent, gl_lock, semaphore, gl_count, num_processes, n_step=10, reward_decay=0.9):
        # self.device = torch.device(
        #     'cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cpu')
        self.gamma = reward_decay
        self.agent = ActorCritic(
            env.observation_space.shape[0], env.action_space.n).double().to(device=self.device)
        self.agent.load_state_dict(gl_agent.state_dict())
        self.gl_optim = optim.Adam(gl_agent.parameters())
        self.optim = optim.Adam(self.agent.parameters())
        self.gl_agent = gl_agent
        self.gl_lock = gl_lock
        self.sem = semaphore
        self.gl_count = gl_count
        self.process_num = num_processes
        self.obs = []
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
        self.obs.append((s, self.latest_log_prob.pop(), r, self.latest_entropy.pop()))
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
        
        self.gl_lock.acquire()
        loss.backward()
        for param, gl_param in zip(self.agent.parameters(), self.gl_agent.parameters()):
            gl_param.grad = param.grad
        self.gl_optim.step()
        self.gl_lock.release()

        with self.gl_count.get_lock():
            self.gl_count.value += 1
            if (self.gl_count.value==self.process_num):
                self.sem.release()
        self.sem.acquire()
        self.sem.release()
        with self.gl_count.get_lock():
            self.gl_count.value -= 1
            if (self.gl_count.value==0):
                self.sem.acquire()

        self.agent.load_state_dict(self.gl_agent.state_dict())

    def convert_to_discounted_reward(self, rewards, next_state_val):
        cumulative_reward = [rewards[-1]+self.gamma*next_state_val]
        for reward in reversed(rewards[:-1]):
            cumulative_reward.append(reward + self.gamma*cumulative_reward[-1])
        return cumulative_reward[::-1]


def worker(gl_agent, gl_r_per_eps, gl_solved, gl_lock, gl_print_lock, semaphore, gl_count, num_processes, worker_num, n_step):
    # env = gym.make("LunarLander-v2")
    env = gym.make("CartPole-v0")
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
    processes = []
    # env = gym.make("LunarLander-v2")
    env = gym.make("CartPole-v0")
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    gl_agent = ActorCritic(
        env.observation_space.shape[0], env.action_space.n).double().to(device=device)
    gl_agent.share_memory()
    gl_r_per_eps: "Queue[int]" = Queue()
    gl_solved = Value('i', False)
    gl_lock = Lock()
    gl_print_lock = Lock()
    semaphore = mp.Semaphore(0)
    gl_count = mp.Value('i', 0)
    num_processes = mp.cpu_count()
    step_size = 128
    for i in range(num_processes):
        p = mp.Process(target=worker, args=(
            gl_agent, gl_r_per_eps, gl_solved, gl_lock, gl_print_lock, 
            semaphore, gl_count, num_processes, i, step_size))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    for p in processes:
        p.terminate()
    reward = []
    while gl_r_per_eps.qsize() != 0:
        reward.append(gl_r_per_eps.get())

    torch.save(gl_agent, 'nstep-A2C.pt')

    del gl_r_per_eps, gl_solved, gl_lock, gl_agent

    r_per_eps_x = [i+1 for i in range(len(reward))]
    r_per_eps_y = reward
    mean_per_100_eps = [np.mean(reward[i:i+100])
                        for i in range(0, len(reward)-100, 100)]
    mean_per_100_eps_x = [(i+1)*100 for i in range(len(mean_per_100_eps))]
    mean_per_100_eps_y = mean_per_100_eps

    plt.plot(r_per_eps_x, r_per_eps_y, mean_per_100_eps_x, mean_per_100_eps_y)
    plt.show()
