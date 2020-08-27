### Interesting Findings about environments:
Since the Frozen Lake Environment only returns a 1 for reaching the Goal state and 0 for falling into a pit, the lack of feedback meant that it takes very very long for the value of the goal state to slowly propagate through the entire path. If falling into the pit returns -1, or if every step returns -0.1, then the agent will be more incentivized to find the shortest and most efficient path.

If we enable slipperiness into the environment, it will surely take a lot longer than it already takes...