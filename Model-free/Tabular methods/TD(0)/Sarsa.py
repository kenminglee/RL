import gym
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import base_agent
from run_experiment import run_experiment

class sarsa_agent(base_agent):
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.1):
        self.alpha = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=[i for i in range(env.env.nA)], dtype=np.float64)

    def


env = gym.make("Taxi-v3")

# print("Action space: ", env.action_space, env.env.nA)
# print("Observation space: ", env.observation_space, env.env.nS)



if __name__ == "__main__":