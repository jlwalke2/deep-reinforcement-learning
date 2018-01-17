from deeprl.agents.DoubleDeepQAgent import DoubleDeepQAgent
from deeprl.memories import PrioritizedMemory
from deeprl.policies import EpsilonGreedyPolicy
from deeprl.utils.async import ModelManager
from deeprl.utils.metrics import *
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import rmsprop
import gym
import multiprocessing
import logging

num_agents = 5
num_episodes = 1000

def run_agent_no_shaping(history):
    env = gym.make('LunarLander-v2')

    num_features = DoubleDeepQAgent._get_space_size(env.observation_space)
    num_actions = DoubleDeepQAgent._get_space_size(env.action_space)

    model = Sequential([
        Dense(64, input_dim=num_features, activation='relu'),
        Dense(64, activation='relu'),
        Dense(units=num_actions, activation='linear')
    ])
    model.compile(loss='mse', optimizer=rmsprop(lr=0.0016, decay=0.000001))
    logger = multiprocessing.log_to_stderr()
    logger.setLevel(logging.INFO)

    agent = DoubleDeepQAgent(name=multiprocessing.current_process().name, env=env, model=model,
                             policy=EpsilonGreedyPolicy(min=0.025, decay=0.96, exploration_episodes=1),
                             memory=PrioritizedMemory(maxlen=50000, sample_size=32),
                             logger=logger, history=history,
                             metrics=[EpisodeReward(), RollingEpisodeReward(), CumulativeReward()],
                             gamma=0.99, max_steps_per_episode=500)

    agent.train(target_model_update=750, max_episodes=num_episodes, render_every_n=num_episodes+1)


def run_agent_with_shaping(history):
    env = gym.make('LunarLander-v2')

    num_features = DoubleDeepQAgent._get_space_size(env.observation_space)
    num_actions = DoubleDeepQAgent._get_space_size(env.action_space)

    model = Sequential([
        Dense(64, input_dim=num_features, activation='relu'),
        Dense(64, activation='relu'),
        Dense(units=num_actions, activation='linear')
    ])
    model.compile(loss='mse', optimizer=rmsprop(lr=0.0016, decay=0.000001))
    logger = multiprocessing.log_to_stderr()
    logger.setLevel(logging.INFO)

    def shape_reward(*args):
        s, a, r, s_prime, done = args
        def potential(state):
            x_pos = state[0]
            y_pos = state[1]
            return (1 - abs(x_pos)) + (1 - y_pos)  # Encourage moving to 0,0 (center of landing pad)

        r = r + 0.99 * potential(s_prime) - potential(s)
        return (s, a, r, s_prime, done)

    agent = DoubleDeepQAgent(name=multiprocessing.current_process().name, env=env, model=model,
                             policy=EpsilonGreedyPolicy(min=0.025, decay=0.96, exploration_episodes=1),
                             memory=PrioritizedMemory(maxlen=50000, sample_size=32),
                             logger=logger, history=history,
                             metrics=[EpisodeReward(), RollingEpisodeReward(), CumulativeReward()],
                             gamma=0.99, max_steps_per_episode=500)
    agent.preprocess_state = shape_reward

    agent.train(target_model_update=750, max_episodes=num_episodes, render_every_n=num_episodes+1)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import os.path
    from warnings import warn
    fname = 'reward_shaping.csv'
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

        metrics.save('reward_shaping.csv')
        df = metrics.get_episode_metrics()
    else:
        import pandas as pd
        warn('Existing history found.  Loading history from {}'.format(fname))
        df = pd.read_csv(fname)
        df.set_index('episode', inplace=True)


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




