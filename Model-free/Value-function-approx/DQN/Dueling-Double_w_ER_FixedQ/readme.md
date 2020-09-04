Algorithm in this folder implements these additional features on top of vanilla DQN:

- Experience replay
- Fixed Q values - parameters are copied from policy net to target net after every training iteration
- Double DQN to reduce maximization bias
- Dueling Architecture - have two separate streams flowing into the output layer of the nn
- Gradient Clipping to prevent exploding/vanishing gradients
