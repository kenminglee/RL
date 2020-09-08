We implement MC here using actor-critic with advantage function. We are not using multiple working processes here.

As a result we should experience the downfall of performing online updates without ER-buffers.

