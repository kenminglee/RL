## Reproducing RL Algorithms

Implementation of different RL algorithms in Pytorch.

Algorithms that were implemented: <br>
Tabular methods: <br>
1. Value Iteration
2. Policy Iteration
3. SARSA
4. Q-learning
5. SARSA lambda
6. Watkins Q-learning

Deep RL:<br>
1. REINFORCE [[code](vanilla_policy_gradient/REINFORCE.py) | [paper](https://papers.nips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf)]
2. DQN [[code](DQN/vanilla_w_ER_FixedQ/cartpole/dqn_er_fixedq.py) | [paper](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)]
3. Dueling Double DQN [[code](DQN/Dueling-Double_w_ER_FixedQ/cartpole/ddqn.py) | [Double DQN paper](https://arxiv.org/pdf/1509.06461.pdf) | [Dueling DQN paper](https://arxiv.org/pdf/1511.06581.pdf)]
4. A3C [[code](A3C/n-stepTD/a3c.py) | [paper](https://arxiv.org/pdf/1602.01783.pdf)]
5. A2C [[MPI implementation](A2C/n-stepTD/a2c_mpi.py) | [Multiprocessing implementation](A2C/n-stepTD/a2c.py) | [blogpost](https://openai.com/blog/baselines-acktr-a2c/)]
6. PPO

Sequence of implementation:

- Model-based: VI and PI
- Model-free Tabular methods: Sarsa -> Q-learning -> Sarsa-lambda -> Watkins Q-learning
- VPG (REINFORCE)
- vanilla DQN with experience replay and fixed Q targets
- Double DQN with Dueling architecture, with experience replay and fixed Q targets
- vanilla actor-critic: TD(0) -> MC -> n-step TD
- A3C: MC -> n-step TD with multiprocessing
- A2C: MC -> n-step TD with multiprocessing
- A2C with MPI and tensorboard


Command to run MPI-enabled scripts
```mpiexec -n <num_processes> python <filename>.py```

Command to run tensorboard-enabled scripts
```tensorboard --logdir=runs```