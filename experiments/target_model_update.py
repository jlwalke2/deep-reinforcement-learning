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

experiment_name = 'target_model_update'
num_agents = 10
num_episodes = 2000



def train_medium_update(agent):
    agent.train(target_model_update=1e-4, max_episodes=num_episodes, render_every_n=None)


def train_slow_update(agent):
    agent.train(target_model_update=1e-6, max_episodes=num_episodes, render_every_n=None)


def build_agent(name):
    env = gym.make('LunarLander-v2')

    num_features = DoubleDeepQAgent._get_space_size(env.observation_space)
    num_actions = DoubleDeepQAgent._get_space_size(env.action_space)

    model = Sequential([
        Dense(64, input_dim=num_features, activation='relu'),
        Dense(64, activation='relu'),
        Dense(units=num_actions, activation='linear')
    ])
    model.compile(loss='mse', optimizer='sgd')

    agent = DoubleDeepQAgent(name=name, env=env, model=model,
                             policy=EpsilonGreedyPolicy(min=0.05, max=0.5, decay=0.999),
                             memory=PrioritizedMemory(maxlen=50000),
                             metrics=[EpisodeReward(), RollingEpisodeReward(), CumulativeReward(), EpisodeTime()],
                             gamma=0.99, max_steps_per_episode=500)

    return agent


if __name__ == '__main__':

    from experiments.experiment import Experiment

    agents = [(build_agent('MediumUpdate'), num_agents, train_medium_update),
              (build_agent('SlowUpdate'), num_agents, train_slow_update)]

    e = Experiment(experiment_name, agents)

    e.run()

    import matplotlib.pyplot as plt
    p = e.get_plot('CumulativeReward')
    plt.show()
# if __name__ == '__main__':
#     import matplotlib.pyplot as plt
#     import os.path
#     from warnings import warn
#     fname = experiment_name + '.h5'
#
#     if not os.path.isfile(fname):
#         manager = ModelManager()
#         manager.start()
#         metrics = manager.Monitor()
#
#         processes = [multiprocessing.Process(target=run_agent_no_shaping, args=(metrics,), name='SoftUpdate_{}'.format(i)) for i in range(num_agents)]
#         processes.extend([multiprocessing.Process(target=run_agent_with_shaping, args=(metrics,), name='HardUpdate_{}'.format(i)) for i in range(num_agents)])
#
#         for p in processes:
#             p.start()
#
#         for p in processes:
#             p.join()
#
#         metrics.save(fname)
#         df = metrics.get_episode_metrics()
#     else:
#         import pandas as pd
#         warn('Existing history found.  Loading history from {}'.format(fname), stacklevel=2)
#         from deeprl.utils import History
#         df, _ =  History().load(fname)
# #        df.set_index('episode', inplace=True)
#
#         # steps_df = pd.read_csv('step_metrics.csv')
#         # steps_df.set_index('step', inplace=True)
#
#     # steps_df.pivot(columns='sender', values='CumulativeReward').groupby(
#     #     lambda colname: 'WithShape' if 'WithShape' in colname else 'NoShape', axis=1).plot()
#     # plt.show()
#     df.pivot(columns='sender', values='CumulativeReward').plot()
#     plt.show()
#
#     fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8,4))
#
#     df.pivot(columns='sender', values='CumulativeReward').groupby(
#         lambda colname: 'SoftUpdate' if 'SoftUpdate' in colname else 'HardUpdate', axis=1).mean().plot(ax=axes[0], title='Mean Cumulative Reward')
#
#     t0 = df.pivot(columns='sender', values='CumulativeReward').groupby(
#         lambda colname: 'SoftUpdate' if 'SoftUpdate' in colname else 'HardUpdate', axis=1)
#
#     means = t0.mean()
#     mins = t0.min()
#     maxs = t0.max()
#
#     for col in means.columns:
#         axes[0].fill_between(means.index, mins[col], maxs[col], alpha=0.2)
#
#     df.pivot(columns='sender', values='EpisodeReward').groupby(
#         lambda colname: 'SoftUpdate' if 'SoftUpdate' in colname else 'HardUpdate', axis=1).mean().plot(ax=axes[1], title='Mean Episode Reward')
#
#     df.pivot(columns='sender', values='EpisodeReward').groupby(
#         lambda colname: 'SoftUpdate' if 'SoftUpdate' in colname else 'HardUpdate', axis=1).mean().rolling(window=50, min_periods=1).mean().plot(ax=axes[1])
#
#     plt.show()
#
#
#     fig.savefig(experiment_name + '_avg_reward.png', dpi=500, linewidth=1)




