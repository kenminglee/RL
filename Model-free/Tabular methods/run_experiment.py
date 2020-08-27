import matplotlib.pyplot as plt
def run_experiment(env, agent, num_eps=50, render_env=False, 
display_plot=True, print_results=True, plot_name=None):
    r_per_eps = []
    for eps in range(num_eps):
        s = env.reset()  
        a = agent.choose_action(s)
        total_r_in_eps = 0
        while True:
            # render env
            if render_env:
                env.render()
            # take action, observe s' and r
            s_, r, done, _ = env.step(a)
            # get next action
            a_ = agent.learn(s, a, r, s_, done)
            total_r_in_eps += r
            s = s_
            a = a_
            if done:
                r_per_eps.append(total_r_in_eps)
                if print_results: 
                    print('Obtained results of', r_per_eps[-1], 'in episode', eps)
                break
    plt.plot(r_per_eps)
    if plot_name:
        plt.savefig(plot_name+'.png')
    if display_plot:
        plt.show()
    return r_per_eps


    