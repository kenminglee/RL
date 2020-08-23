def run_experiment(env, agent, num_eps=50, render_env=False, display_plot=True, print_results=True):
    r_per_eps = []
    for eps in range(num_eps):
        env.reset()  
        s = env.env.state
        a = agent.choose_action(s)
        total_r_in_eps = 0
        while True:
            # render env
            if render_env:
                env.render()
            # take action, observe s' and r
            s_, r, done, _ = env.step(a)
            # get next action
            a = agent.learn(s, a, r, s_, done)
            total_r_in_eps += r
            s = s_
            if done:
                r_per_eps.append(total_r_in_eps)
                if print_results: 
                    print('Obtained results of', r_per_eps[-1], 'in episode', eps)
                break
