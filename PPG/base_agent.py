class base_agent:
    def choose_action(self, s) -> int:
        """Get an action from our agent based on the observed state s

            Parameters:
            s: state

            Returns:
            int: action to take

        """
        raise NotImplementedError('choose_action function not implemented')

    def learn(self, s, a, r, s_, done) -> int:
        """Learn based on S, A, R and S', then return the next action to take

            Parameters:
            s: state
            a: action taken in state s
            r: reward observed by taking (s,a)
            s_: next state

            Returns:
            int: action to take for state s_
        """
        raise NotImplementedError('learn function not implemented')