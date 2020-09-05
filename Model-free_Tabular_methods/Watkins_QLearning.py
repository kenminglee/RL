import gym
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from base_agent import base_agent
from run_experiment import run_experiment

class watkins_qlearning(base_agent):
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.1, eligibility_trace_decay=0.9, 
    eligibility_trace_update_threshold=0.1):
        self.actions = actions
        self.alpha = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.lambda_ = eligibility_trace_decay
        self.theta = eligibility_trace_update_threshold
        self.q_table = pd.DataFrame(columns=actions, dtype=np.float64)
        self.e_table = pd.DataFrame(columns=actions, dtype=np.float64)

    def choose_action(self, s) -> int:
        self.check_state_exist(s)
        if np.random.uniform() >= self.epsilon:
            state_action = self.q_table.loc[s, :]
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            action = np.random.choice(self.actions)
        return action 

    def learn(self, s, a, r, s_, done) -> int:
        a_ = self.choose_action(s_)
        a_star = a_ if self.q_table.loc[s_, a_]==np.max(self.q_table.loc[s_,:]) else None
        delta = r + self.gamma * self.q_table.loc[s_, a_] - self.q_table.loc[s, a]
        self.e_table.loc[s, a] += 1
        for state in self.q_table.index:
            for action in self.q_table.columns:
                if self.e_table.loc[state, action] < self.theta: continue
                self.q_table.loc[state, action] += self.alpha * delta * self.e_table.loc[state, action]
                if a_star==a_:
                    self.e_table.loc[state, action] *= (self.gamma * self.lambda_)
                else: 
                    self.e_table.loc[state, action] = 0
        if done:
            self.reset_all_traces()
        return a_
    
    def reset_all_traces(self):
        for state in self.e_table.index:
            self.e_table.loc[state, :] = [0 for _ in range(len(self.actions))]

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0 for _ in range(len(self.actions))],
                    index=self.q_table.columns,
                    name=state,
                )
            )
            self.e_table = self.e_table.append(
                pd.Series(
                    [0 for _ in range(len(self.actions))],
                    index=self.e_table.columns,
                    name=state,
                )
            )

if __name__ == "__main__":
    env = gym.make("Taxi-v3")
    print("Action space: ", env.action_space, env.env.nA)
    print("Observation space: ", env.observation_space, env.env.nS)
    
    # no epsilon decay over time
    agent = watkins_qlearning([i for i in range(env.env.nA)])

    run_experiment(env, agent, num_eps=2500, render_env=False)


