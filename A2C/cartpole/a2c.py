from base_agent import base_agent
import torch.nn as nn
import torch
import torch.optim as optim
from torch.distributions import Categorical
import torch.multiprocessing as mp
import torch.nn.functional as F
import gym
import numpy as np
import matplotlib.pyplot as plt

# A3C paper: https://arxiv.org/pdf/1602.01783.pdf
# A2C blogpost: https://openai.com/blog/baselines-acktr-a2c/

class ActorCritic(nn.Module):
    def __init__(self, input_size, n_actions):
        super(ActorCritic, self).__init__()
        self.first_layer = nn.Linear(input_size, 128)
        self.critic_layer = nn.Linear(128, 64)
        self.critic_head = nn.Linear(64, 1)
        self.actor_head = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.first_layer(x))
        return self.actor_head(x), self.critic_head(F.relu(self.critic_layer(x.detach())))


class Agent(base_agent):
    def __init__(self, agent, n_step=10, reward_decay=0.9):
        self.device = torch.device('cpu')
        self.gamma = reward_decay
        self.agent = ActorCritic(
            env.observation_space.shape[0], env.action_space.n).double().to(device=self.device)
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
        states, log_probs, rewards, entropy = zip(*self.obs)
        states = torch.tensor(states, dtype=torch.double).to(
            device=self.device)
        if not done:
            _, val = self.agent(torch.tensor(
                s_, dtype=torch.double).to(device=self.device))
            discounted_rewards = self.convert_to_discounted_reward(
                rewards, val)
        else:
            discounted_rewards = self.convert_to_discounted_reward(rewards, 0)
        discounted_rewards = torch.tensor(
            discounted_rewards, device=self.device)
        log_probs = torch.stack(
            log_probs).to(device=self.device)
        entropy = torch.stack(entropy).to(device=self.device)
        _, critic_values = self.agent(states)
        delta = discounted_rewards - critic_values
        loss = torch.mean(-log_probs*(delta.detach()) - 0.001*entropy + 0.1*(delta**2))

        self.gl_lock.acquire()
        self.gl_optim.zero_grad()
        loss.backward()
        for param, gl_param in zip(self.agent.parameters(), self.gl_agent.parameters()):
            gl_param.grad = param.grad
        self.gl_optim.step()
        self.gl_lock.release()

        self.agent.load_state_dict(self.gl_agent.state_dict())

    def convert_to_discounted_reward(self, rewards, next_state_val):
        cumulative_reward = [rewards[-1]+self.gamma*next_state_val]
        for reward in reversed(rewards[:-1]):
            cumulative_reward.append(reward + self.gamma*cumulative_reward[-1])
        return cumulative_reward[::-1]
    
    def load_state_dict(self, state_dict):
        self.agent.load_state_dict(state_dict)

def run(gl_return, gl_agent, gl_agent_lock, gl_print_lock, gl_count, semaphore, worker_num, gl_solved):
    env = gym.make("CartPole-v0")
    # env = gym.wrappers.Monitor(env, "recording", force=True)
    device = torch.device('cpu')
    network = ActorCritic(
            env.observation_space.shape[0], env.action_space.n).double().to(device=self.device)
    agent = Agent(network, gl_agent)
    r_per_eps = []
    
    while not bool(gl_solved.value):
        s = env.reset()
        a = agent.choose_action(s)
        total_r_in_eps = 0
        while True:
            s_, r, done, _ = env.step(a)
            a_ = agent.learn(s, a, r, s_, done)
            total_r_in_eps += r
            s, a = s_, a_
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
    
    


def training_loop():
    env = gym.make("CartPole-v0")
    device = torch.device('cpu')
    gl_agent = ActorCritic(
        env.observation_space.shape[0], env.action_space.n).double().to(device=device)

    num_processes = mp.cpu_count()
    gl_agent.share_memory()
    gl_agent_lock = mp.Lock()
    gl_print_lock = mp.Lock()
    gl_ret: "Queue[Dict]" = mp.Queue()
    semaphore = mp.Semaphore(num_processes)
    gl_count = mp.Value('i', 0)
    gl_solved = Value('i', False)
    
    processes = []
    reward = []
    for i in range(num_processes):
        p = mp.Process(target=run, args=(gl_ret, gl_agent, gl_agent_lock, gl_print_lock, gl_count, semaphore, i, gl_solved))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    for p in processes:
        p.terminate()
    ret_list = []
    while gl_ret.qsize()!=0:
        ret_list.append(gl_ret.get())

    torch.save(gl_agent, 'nStep-A2C.pt')
    del gl_agent, gl_ret
    r_per_eps_x = [i+1 for i in range(len(reward))]
    r_per_eps_y = reward

    plt.plot(r_per_eps_x, r_per_eps_y)
    plt.show()


if __name__ == "__main__":


    

    
