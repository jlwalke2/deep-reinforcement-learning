import gym
from deeprl.agents.ac3 import A3CAgent
import keras.backend as K
from keras.layers import Dense, Flatten, Input, concatenate
from keras.models import Sequential, Model
from keras.optimizers import rmsprop

config = dict(version=1,
              handlers={'console': {'class': 'logging.StreamHandler', 'level': 'INFO'}},
              root={'level': 'DEBUG', 'handlers': ['console']},
              disable_existing_loggers=False)

import logging.config
logging.config.dictConfig(config)

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')

    num_features = env.observation_space.shape[0]
    num_actions = env.action_space.n

    actor = Sequential([
        Dense(32, input_dim=num_features, activation='relu'),
        Dense(32, activation='relu'),
        Dense(units=num_actions, activation='softmax')
    ])
    actor.compile(loss='mse', optimizer=rmsprop(lr=0.0016, decay=0.000001))

    critic = Sequential([
        Dense(32, input_dim=num_features, activation='relu'),
        Dense(32, activation='relu'),
        Dense(units=1, activation='linear')
    ])
    critic.compile(loss='mse', optimizer=rmsprop(lr=0.0016, decay=0.000001))

    agent = A3CAgent(env=env, actor=actor, critic=critic, max_steps_per_episode=500)
    agent.train(num_workers=4, max_episodes=500, render_every_n=10)

    df = agent.history.get_episode_metrics()
    if df.shape[0] > 0:
        p = df.plot.line(x='episode_count', y='total_reward')
        p = df.plot.line(x='episode_count', y='mean_reward', ax=p)
        p.figure.show()
    pass