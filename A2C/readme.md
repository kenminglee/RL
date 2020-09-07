Recall that DQN uses experience-replay buffer to decorrelate data, and performing gradient descent in batches reduces the variance. But as a result any implementation that requires the use of ER-buffer must go off-policy.

Actor-critic methods (A2C and A3C) replaces the need of ER-buffers by spawning multiple worker processes who work independently and therefore experience very different states. This allows us to also have de-correlated data and allow us to train in batches. Since we do not need ER-buffers any more, A2C & A3C etc. can use on-policy training (as well as off-policy) algorithms. MC, TD(0), n-step TD, or even TD(lambda) works with actor-critic methods.

The difference between A2C and A3C