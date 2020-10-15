Useful resources:

Original paper: 
https://arxiv.org/pdf/1707.06347.pdf

Blogpost:
https://openai.com/blog/openai-baselines-ppo/

Amazing explaination of PPO:
https://stackoverflow.com/a/50663200

SpinningUp:
https://spinningup.openai.com/en/latest/algorithms/ppo.html

The pseudocode from Spinning Up might be a little misleading as it doesn't show the part where multiple gradient ascend can be performed per update (the actual implementation does tho...). Refer to stackoverflow's explaination for details on the key parts of PPO.

Spinning up's implementation normalizes the advantage reward, performs SGA on entire batch for x iterations - and stops if KL divergence [0, inf) between new and old policy becomes greater than 1.5!