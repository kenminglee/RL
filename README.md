# Reproducing RL Algorithms

<!--- > `dev` branch may contain work in progress. For stable code, checkout `master` branch instead.  --->

A short writeup about the deep RL algorithms implemented here [[pdf](writeups/understanding_ppg.pdf) | [tex](writeups/understanding_ppg.tex)]

## Algorithms that were implemented:
Code referred to below are implementations in this repository, paper refers to original publications of these algorithms.

Tabular methods: <br>
1. Value Iteration and Policy Iteration [[VI code](Model-based/value-iteration.py) | [PI code](Model-based/policy-iteration.py) | [paper](https://www.ics.uci.edu/~dechter/publications/r42a-mdp_report.pdf)]
2. SARSA [[code](Model-free_Tabular_methods/Sarsa.py) | [paper](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.17.2539&rep=rep1&type=pdf)]
3. Q-learning [[code](Model-free_Tabular_methods/Q-Learning.py) | [paper](http://www.cs.rhul.ac.uk/~chrisw/new_thesis.pdf)]
4. SARSA lambda [[code](Model-free_Tabular_methods/Sarsa-lambda.py)]
5. Watkins Q-learning [[code](Model-free_Tabular_methods/Watkins_QLearning.py)]

Deep RL:<br>
Discrete: <br>
1. REINFORCE [[code](vanilla_policy_gradient/REINFORCE.py) | [paper](https://papers.nips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf)]
2. DQN [[code](DQN/vanilla_w_ER_FixedQ/cartpole/dqn_er_fixedq.py) | [paper](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)]
3. Dueling Double DQN [[code](DQN/Dueling-Double_w_ER_FixedQ/cartpole/ddqn.py) | [Double DQN paper](https://arxiv.org/pdf/1509.06461.pdf) | [Dueling DQN paper](https://arxiv.org/pdf/1511.06581.pdf)]
4. A3C [[code](A3C/n-stepTD/a3c.py) | [paper](https://arxiv.org/pdf/1602.01783.pdf)]
5. A2C [[MPI implementation](A2C/n-stepTD/a2c_mpi.py) | [Multiprocessing implementation](A2C/n-stepTD/a2c.py) | [blogpost](https://openai.com/blog/baselines-acktr-a2c/)]
6. PPO [[code](PPO/ppo.py) | [paper](https://arxiv.org/pdf/1707.06347.pdf)]
7. PPG (Work in progress) [[code](PPG/ppg.py) | [paper](https://arxiv.org/pdf/2009.04416.pdf)]

Continuous: <br>
1. DDPG [[code](DDPG/ddpg.py) | [paper](https://arxiv.org/pdf/1509.02971.pdf)]
## Implementation Results
### Discrete
These algorithms ran on OpenAI Gym's [Cartpole environment](https://gym.openai.com/envs/CartPole-v0/)
####  REINFORCE 
<img src="vanilla_policy_gradient/Solved_cartpole_1100eps.png" width="800"/>

#### DQN
<img src="DQN/old_impl/vanilla_w_ER_FixedQ/cartpole/Solved-cartpole-1800eps.png" width="800">

#### Dueling DDQN
<img src="DQN/DDQN_clean/duelingddqn_cartpole.png" width="800">

#### A3C
<img src="A3C/cartpole-MC/cartpole_800eps.png" width="800">

### Continuous
These algorithms ran on OpenAI Gym's [Pendulum environment](https://gym.openai.com/envs/Pendulum-v0/)

#### DDPG
Mean reward over past 100 episode of -169.5 after 1000 episodes
<img src="DDPG/ddpg_pendulum.png" width="800"/>
<!-- <figure>
    <img src="DDPG/ddpg_pendulum.png" width="800"/>
    <figcaption>Mean reward over past 100 episode of -169.5 after 1000 episodes</figcaption>
</figure> -->

#### TD3
Mean reward over past 100 episode of -174.8 after 1200 episodes. Notice the lower variance compared to DDPG above.
<img src="TD3/td3_pendulum.png" width="800"/>

## Commands to Run Scripts
For all non-MPI enabled / algorithms that do not require parallel workers, run the scripts with <br>
``` python3 <filename>```

Command to run MPI-enabled scripts <br>
```mpiexec -n <num_processes> python <filename>.py```


Command to run tensorboard-enabled scripts <br>
```tensorboard --logdir=runs```, <br>
where `runs` is the default folder that tensorboard writes data to

<!-- ## Sequence of implementation:
- Model-based: VI and PI
- Model-free Tabular methods: Sarsa -> Q-learning -> Sarsa-lambda -> Watkins Q-learning
- VPG (REINFORCE)
- vanilla DQN with experience replay and fixed Q targets
- Double DQN with Dueling architecture, with experience replay and fixed Q targets
- vanilla actor-critic: TD(0) -> MC -> n-step TD
- A3C: MC -> n-step TD with multiprocessing
- A2C: MC -> n-step TD with multiprocessing
- A2C with MPI and tensorboard -->



