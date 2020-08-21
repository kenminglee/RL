import gym
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("FrozenLake-v0", is_slippery=True, map_name="8x8")
# env.reset()                    
# env.render()

# print("Action space: ", env.action_space, env.env.nA)
# print("Observation space: ", env.observation_space, env.env.nS)
# print("Transition probabilities: ", env.env.P[15])
#  env.env.P[15] returns {0: [(1.0, 15, 0, True)], 1: [(1.0, 15, 0, True)], 2: [(1.0, 15, 0, True)], 3: [(1.0, 15, 0, True)]}
# [Pr(next_state), next_state, reward, is_terminal]

# Create a Q-table with # rows equals # states, and # cols equals # actions
q_table = pd.DataFrame([(0,0,0,0) for _ in range(env.env.nS)], columns=[i for i in range(env.env.nA)], dtype=np.float64)

# Param
theta = 0.0001
gamma = 0.9

# Value iteration loop
while True:
    delta = 0
    for state in range(env.env.nS):
        for action in range(env.env.nA):
            old_val = q_table.loc[state, action]
            sum_ = 0
            for prob, s_, r, is_terminal in env.env.P[state][action]:
                max_next_state_q = max(q_table.loc[s_])
                sum_ += prob * (r + gamma*max_next_state_q)
            # updating values in place is totally okay, don't have to create another q_table
            q_table.loc[state, action] = sum_
            delta = max(delta, abs(old_val - q_table.loc[state, action]))

    if delta < theta: 
        break

print('converged!')
# print(q_table)

# Let agent interact with env
num_of_episodes = 1000
r_per_eps = []
for i in range(num_of_episodes):
    env.reset()  
    s = 0
    total_r_in_eps = 0
    while True:
        # render env
        # env.render()
        # pick action
        state_action = q_table.loc[s,]
        a = np.random.choice(state_action[state_action==np.max(state_action)].index)
        # take action, observe s' and r
        s_, r, done, _ = env.step(a)
        # prep for next step
        total_r_in_eps += r
        s = s_
        if done:
            r_per_eps.append(total_r_in_eps)
            break


# Plot results
plt.plot(r_per_eps)
plt.show()