import matplotlib.pyplot as plt
import gym
import numpy as np


def run_experiment(env, agent, num_eps=50, display_plot=True, plot_name=None, save_checkpoint_exist=False, checkpoint_name='model', save_every_x_eps=1000, r_per_eps=None, initial_eps=0):
    r_per_eps = [] if not r_per_eps else r_per_eps
    assert(len(r_per_eps)==initial_eps)
    mean_per_100_eps = []
    for eps in range(initial_eps, initial_eps+num_eps):
        s = env.reset()
        a = agent.choose_action(s)
        total_r_in_eps = 0
        while True:
            # take action, observe s' and r
            s_, r, done, _ = env.step(a)
            # get next action
            a_ = agent.learn(s, a, r, s_, done)
            total_r_in_eps += r
            s = s_
            a = a_
            if done:
                r_per_eps.append(total_r_in_eps)
                if len(r_per_eps) % 100 == 0:
                    mean = np.mean(r_per_eps[-100:])
                    mean_per_100_eps.append(mean)
                    print('Average reward for past 100 episode',
                          mean, 'in episode', eps+1)
                if save_checkpoint_exist and len(r_per_eps)%save_every_x_eps==0:
                    agent.save_checkpoint(checkpoint_name+'_'+str(eps+1)+'.tar', eps+1, r_per_eps)
                break
    r_per_eps_x = [i+1 for i in range(len(r_per_eps))]
    r_per_eps_y = r_per_eps

    mean_per_100_eps_x = [(i+1)*100 for i in range(len(mean_per_100_eps))]
    mean_per_100_eps_y = mean_per_100_eps

    plt.plot(r_per_eps_x, r_per_eps_y, mean_per_100_eps_x, mean_per_100_eps_y)
    if plot_name:
        plt.savefig(plot_name+'.png')
    if display_plot:
        plt.show()
    return r_per_eps


def run_cartpole_experiment(agent, display_plot=True, render_env=False, print_results=True, plot_name=None):
    env = gym.make("CartPole-v0")
    env = gym.wrappers.Monitor(env, "recording", force=True)
    r_per_eps = []
    mean_per_100_eps = []
    solved = False
    eps = 0
    while not solved:
        eps += 1
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
                if len(r_per_eps) % 100 == 0:
                    mean = np.mean(r_per_eps[-100:])
                    mean_per_100_eps.append(mean)
                    print('Average reward for past 100 episode',
                          mean, 'in episode', eps)
                    if mean >= 195:
                        solved = True
                        print('Solved at episode', eps)
                break

    r_per_eps_x = [i+1 for i in range(len(r_per_eps))]
    r_per_eps_y = r_per_eps

    mean_per_100_eps_x = [(i+1)*100 for i in range(len(mean_per_100_eps))]
    mean_per_100_eps_y = mean_per_100_eps

    plt.plot(r_per_eps_x, r_per_eps_y, mean_per_100_eps_x, mean_per_100_eps_y)
    if plot_name:
        plt.savefig(plot_name+'.png')
    if display_plot:
        plt.show()
    return r_per_eps
