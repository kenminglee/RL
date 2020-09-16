from base_agent import base_agent
import torch.nn as nn
import torch
import torch.optim as optim
import torch.multiprocessing as mp
from torch.distributions import Categorical
from torch.multiprocessing import Queue, Lock, Value
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
    def __init__(self, env, gl_agent, gl_lock, n_step=10, reward_decay=0.9):
        # self.device = torch.device(
        #     'cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cpu')
        self.gamma = reward_decay
        self.agent = ActorCritic(
            env.observation_space.shape[0], env.action_space.n).double().to(device=self.device)
        self.agent.load_state_dict(gl_agent.state_dict())
        self.gl_optim = optim.Adam(gl_agent.parameters())
        self.gl_agent = gl_agent
        self.gl_lock = gl_lock
        self.obs = []
        self.n_step = n_step
        self.latest_log_prob = []

    def choose_action(self, s) -> int:
        s = torch.tensor(s, dtype=torch.double).to(device=self.device)
        actor_logits, _ = self.agent(s)
        probs = Categorical(logits=actor_logits)
        a = probs.sample()
        assert len(self.latest_log_prob) == 0
        self.latest_log_prob.append(probs.log_prob(a))
        return a.tolist()

    def learn(self, s, a, r, s_, done) -> int:
        self.obs.append((s, self.latest_log_prob.pop(), r,))
        if done:
            self.perform_learning_iter(s_, done)
            self.obs = []
            return 0
        return self.choose_action(s_)

    def perform_learning_iter(self, s_, done):
        states, log_probs, rewards = zip(*self.obs)
        states = torch.tensor(states, dtype=torch.double).to(
            device=self.device)
        if not done:
            rewards = list(rewards)
            _, val = self.agent(s_)
            rewards[-1] += self.gamma*val
        discounted_rewards = self.convert_to_discounted_reward(rewards)
        discounted_rewards = torch.tensor(
            discounted_rewards, device=self.device)
        log_probs = torch.stack(
            log_probs).to(device=self.device)
        _, critic_values = self.agent(states)
        delta = discounted_rewards - critic_values
        loss = torch.sum(-log_probs*(delta.detach())) + \
            0.1*torch.mean(delta**2)

        self.gl_lock.acquire()
        self.gl_optim.zero_grad()
        loss.backward()
        for param, gl_param in zip(self.agent.parameters(), self.gl_agent.parameters()):
            gl_param.grad = param.grad
        self.gl_optim.step()
        self.gl_lock.release()

        self.agent.load_state_dict(self.gl_agent.state_dict())

    def convert_to_discounted_reward(self, rewards):
        cumulative_reward = [rewards[-1]]
        for reward in reversed(rewards[:-1]):
            cumulative_reward.append(reward + self.gamma*cumulative_reward[-1])
        return cumulative_reward[::-1]


def worker(gl_agent, gl_r_per_eps, gl_solved, gl_lock, gl_print_lock, worker_num, n_step):
    env = gym.make("CartPole-v0")
    # env = gym.wrappers.Monitor(env, "recording", force=True)
    agent = Agent(env, gl_agent, gl_lock, n_step=n_step)
    r_per_eps = []
    while not bool(gl_solved.value):
        s = env.reset()
        a = agent.choose_action(s)
        total_r_in_eps = 0
        while True:
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
                        gl_solved.value = True
                        gl_print_lock.acquire()
                        print(worker_num, ': Solved at episode', len(r_per_eps))
                        gl_print_lock.release()
                break


if __name__ == "__main__":
    processes = []
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
    for i in range(mp.cpu_count()):
        p = mp.Process(target=worker, args=(
            gl_agent, gl_r_per_eps, gl_solved, gl_lock, gl_print_lock, i, 1))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    for p in processes:
        p.terminate()
    reward = []
    while gl_r_per_eps.qsize() != 0:
        reward.append(gl_r_per_eps.get())

    torch.save(gl_agent, 'MC-A3C.pt')

    del gl_r_per_eps, gl_solved, gl_lock, gl_agent

    r_per_eps_x = [i+1 for i in range(len(reward))]
    r_per_eps_y = reward
    mean_per_100_eps = [np.mean(reward[i:i+100])
                        for i in range(0, len(reward)-100, 100)]
    mean_per_100_eps_x = [(i+1)*100 for i in range(len(mean_per_100_eps))]
    mean_per_100_eps_y = mean_per_100_eps

    plt.plot(r_per_eps_x, r_per_eps_y, mean_per_100_eps_x, mean_per_100_eps_y)
    plt.show()
