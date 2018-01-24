from deeprl.agents.DoubleDeepQAgent import DoubleDeepQAgent
from deeprl.memories import PrioritizedMemory
from deeprl.policies import EpsilonGreedyPolicy
from deeprl.utils.metrics import *
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import adam
import gym
import numpy as np


num_episodes = 10

def build_agent_no_shaping():
    env = gym.make('MountainCar-v0')

    num_features = DoubleDeepQAgent._get_space_size(env.observation_space)
    num_actions = DoubleDeepQAgent._get_space_size(env.action_space)

    model = Sequential()
    model.add(Dense(16, input_dim=num_features, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(units=num_actions, activation='linear'))
    model.compile(loss='mse', optimizer=adam())

    agent = DoubleDeepQAgent(name='AgentNoShaping', env=env, model=model,
                             policy=EpsilonGreedyPolicy(min=0.05, max=0.5, decay=0.999),
                             memory=PrioritizedMemory(maxlen=50000),
                             metrics=[EpisodeReward(), RollingEpisodeReward(), CumulativeReward(), EpisodeTime()],
                             gamma=0.99, max_steps_per_episode=1000)

    return agent

def run_agent(agent):
    agent.train(target_model_update=1e-2, max_episodes=num_episodes, render_every_n=None)


def shape_reward(*args):
    # state is (position, velocity)
    s, a, r, s_prime, done = args

    def potential(state):
        return 1. if np.all(state > 0) or np.all(state < 0) else 0

    r = r + 0.99 * potential(s_prime) - potential(s)

    return args

def build_agent_with_shaping():
    env = gym.make('MountainCar-v0')

    num_features = DoubleDeepQAgent._get_space_size(env.observation_space)
    num_actions = DoubleDeepQAgent._get_space_size(env.action_space)


    model = Sequential()
    model.add(Dense(16, input_dim=num_features, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(units=num_actions, activation='linear'))
    model.compile(loss='mse', optimizer=adam())

    agent = DoubleDeepQAgent(name='AgentWithShaping', env=env, model=model,
                             policy=EpsilonGreedyPolicy(min=0.05, max=0.5, decay=0.999),
                             memory=PrioritizedMemory(maxlen=50000),
                             metrics=[EpisodeReward(), RollingEpisodeReward(), CumulativeReward(), EpisodeTime()],
                             gamma=0.99, max_steps_per_episode=1000)
    agent.preprocess_state = shape_reward

    return agent


if __name__ == '__main__':

    from experiments.experiment import Experiment

    agents = [(build_agent_no_shaping(), 3, run_agent),
              (build_agent_with_shaping(), 3, run_agent)]

    e = Experiment('reward_shaping', agents)

    e.run()

    # steps_df.pivot(columns='sender', values='CumulativeReward').groupby(
    #     lambda colname: 'WithShape' if 'WithShape' in colname else 'NoShape', axis=1).plot()
    # plt.show()

#    df = e.get_plots([])

    # df.pivot(columns='sender', values='CumulativeReward').plot()
    # plt.show()
    #
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8,4))

    e.get_plot('CumulativeReward')
    e.get_plot('EpisodeReward')

    plt.show()
    pass
    # t0 = df.pivot(columns='sender', values='CumulativeReward').groupby(
    #     lambda colname: 'WithShape' if 'WithShape' in colname else 'NoShape', axis=1)
    #
    # means = t0.mean()
    # mins = t0.min()
    # maxs = t0.max()
    #
    # for col in means.columns:
    #     axes[0].fill_between(means.index, mins[col], maxs[col], alpha=0.2)
    #
    # df.pivot(columns='sender', values='EpisodeReward').groupby(
    #     lambda colname: 'WithShape' if 'WithShape' in colname else 'NoShape', axis=1).mean().plot(ax=axes[1], title='Mean Episode Reward')
    #
    # df.pivot(columns='sender', values='EpisodeReward').groupby(
    #     lambda colname: 'WithShape' if 'WithShape' in colname else 'NoShape', axis=1).mean().rolling(window=50, min_periods=1).mean().plot(ax=axes[1])
    #
    # plt.show()
    #
    #
    # fig.savefig('reward_shaping_avg_reward.png', dpi=500, linewidth=1)




