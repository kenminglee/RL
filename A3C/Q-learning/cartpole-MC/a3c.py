from base_agent import base_agent
from run_experiment import run_cartpole_experiment
import torch.nn as nn
import torch
import torch.optim as optim
import torch.multiprocessing as mp
from torch.multiprocessing import Queue, Lock, Value
import torch.nn.functional as F
import gym

# A3C paper: https://arxiv.org/pdf/1602.01783.pdf


class ActorCritic(nn.Module):
    def __init__(self, input_size, n_actions):
        super(Critic, self).__init__()
        self.first_layer = nn.Linear(input_size, 128)
        self.critic_head = nn.Linear(128, 1)
        self.actor_head = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.first_layer(x))
        return self.actor_head(x), self.critic_head(x.detach())


def worker(agent, gl_r_per_eps, gl_solved, gl_lock):
    env = gym.make("CartPole-v0")
    # env = gym.wrappers.Monitor(env, "recording", force=True)
    r_per_eps = []
    while not bool(gl_solved):
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
                    gl_lock.acquire()
                    print('Average reward for past 100 episode',
                          mean, 'in episode', len(r_per_eps))
                    gl_lock.release()
                    if mean >= 195:
                        gl_solved = True
                        gl_lock.acquire()
                        print('Solved at episode', len(r_per_eps))
                        gl_lock.release()
                break


if __name__ == "__main__":
    processes = []
    env = gym.make("CartPole-v0")
    gl_agent = ActorCritic(env.observation_space.shape[0], env.action_space.n)
    gl_agent.share_memory()
    gl_r_per_eps: Queue[int] = Queue()
    gl_solved = Value('i', False)
    gl_lock = Lock()
    proc_num = mp.cpu_count()
    for _ in range(proc_num):
        p = mp.Process(target=worker)
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    for p in processes:
        p.terminate()
    reward = []
    while gl_r_per_eps.qsize() != 0:
        reward.append(gl_r_per_eps.get())
    del gl_r_per_eps, gl_solved, gl_lock, gl_agent
