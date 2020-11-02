# RL

Implementation of different RL algorithms

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
'''mpiexec -n <num_processes> python <filename>.py'''

Command to run tensorboard-enabled scripts
```tensorboard --logdir=runs```