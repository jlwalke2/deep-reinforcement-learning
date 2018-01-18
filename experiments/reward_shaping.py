from deeprl.agents.DoubleDeepQAgent import DoubleDeepQAgent
from deeprl.memories import PrioritizedMemory
from deeprl.policies import EpsilonGreedyPolicy
from deeprl.utils.async import ModelManager
from deeprl.utils.metrics import *
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import adam
import gym
import multiprocessing
import logging
import numpy as np

num_agents = 5
num_episodes = 1000

def run_agent_no_shaping(history):
    env = gym.make('MountainCar-v0')

    num_features = DoubleDeepQAgent._get_space_size(env.observation_space)
    num_actions = DoubleDeepQAgent._get_space_size(env.action_space)

    model = Sequential()
    model.add(Dense(16, input_dim=num_features, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(units=num_actions, activation='linear'))
    model.compile(loss='mse', optimizer=adam())

    logger = multiprocessing.log_to_stderr()
    logger.setLevel(logging.INFO)

    agent = DoubleDeepQAgent(name=multiprocessing.current_process().name, env=env, model=model,
                             policy=EpsilonGreedyPolicy(min=0.05, max=0.5, decay=0.999),
                             memory=PrioritizedMemory(maxlen=50000),
                             logger=logger, history=history,
                             metrics=[EpisodeReward(), RollingEpisodeReward(), CumulativeReward(), EpisodeTime()],
                             gamma=0.99, max_steps_per_episode=1000)

    agent.train(target_model_update=1e-2, max_episodes=num_episodes, render_every_n=None)


def run_agent_with_shaping(history):
    env = gym.make('MountainCar-v0')

    num_features = DoubleDeepQAgent._get_space_size(env.observation_space)
    num_actions = DoubleDeepQAgent._get_space_size(env.action_space)

    def shape_reward(*args):
        # state is (position, velocity)
        s, a, r, s_prime, done = args

        def potential(state):
            return 1. if np.all(state > 0) or np.all(state < 0) else 0

        r = r + 0.99 * potential(s_prime) - potential(s)

        return args

    model = Sequential()
    model.add(Dense(16, input_dim=num_features, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(units=num_actions, activation='linear'))
    model.compile(loss='mse', optimizer=adam())

    logger = multiprocessing.log_to_stderr()
    logger.setLevel(logging.INFO)

    agent = DoubleDeepQAgent(name=multiprocessing.current_process().name, env=env, model=model,
                             policy=EpsilonGreedyPolicy(min=0.05, max=0.5, decay=0.999),
                             memory=PrioritizedMemory(maxlen=50000),
                             logger=logger, history=history,
                             metrics=[EpisodeReward(), RollingEpisodeReward(), CumulativeReward(), EpisodeTime()],
                             gamma=0.99, max_steps_per_episode=1000)
    agent.preprocess_state = shape_reward

    agent.train(target_model_update=1e-2, max_episodes=num_episodes, render_every_n=None)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import os.path
    from warnings import warn
    fname = 'reward_shaping.h5'

    if not os.path.isfile(fname):
        manager = ModelManager()
        manager.start()
        metrics = manager.Monitor()

        processes = [multiprocessing.Process(target=run_agent_no_shaping, args=(metrics,), name='AgentNoShape_{}'.format(i)) for i in range(num_agents)]
        processes.extend([multiprocessing.Process(target=run_agent_with_shaping, args=(metrics,), name='AgentWithShape_{}'.format(i)) for i in range(num_agents)])

        for p in processes:
            p.start()

        for p in processes:
            p.join()

        metrics.save(fname)
        df = metrics.get_episode_metrics()
    else:
        import pandas as pd
        warn('Existing history found.  Loading history from {}'.format(fname), stacklevel=2)
        from deeprl.utils import History
        df, _ =  History().load(fname)
        df.set_index('episode', inplace=True)

        # steps_df = pd.read_csv('step_metrics.csv')
        # steps_df.set_index('step', inplace=True)

    # steps_df.pivot(columns='sender', values='CumulativeReward').groupby(
    #     lambda colname: 'WithShape' if 'WithShape' in colname else 'NoShape', axis=1).plot()
    # plt.show()
    df.pivot(columns='sender', values='CumulativeReward').plot()
    plt.show()

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8,4))

    df.pivot(columns='sender', values='CumulativeReward').groupby(
        lambda colname: 'WithShape' if 'WithShape' in colname else 'NoShape', axis=1).mean().plot(ax=axes[0], title='Mean Cumulative Reward')

    t0 = df.pivot(columns='sender', values='CumulativeReward').groupby(
        lambda colname: 'WithShape' if 'WithShape' in colname else 'NoShape', axis=1)

    means = t0.mean()
    mins = t0.min()
    maxs = t0.max()

    for col in means.columns:
        axes[0].fill_between(means.index, mins[col], maxs[col], alpha=0.2)

    df.pivot(columns='sender', values='EpisodeReward').groupby(
        lambda colname: 'WithShape' if 'WithShape' in colname else 'NoShape', axis=1).mean().plot(ax=axes[1], title='Mean Episode Reward')

    df.pivot(columns='sender', values='EpisodeReward').groupby(
        lambda colname: 'WithShape' if 'WithShape' in colname else 'NoShape', axis=1).mean().rolling(window=50, min_periods=1).mean().plot(ax=axes[1])

    plt.show()


    fig.savefig('reward_shaping_avg_reward.png', dpi=500, linewidth=1)




