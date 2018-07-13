import gym
from deeprl.agents.a3c import A3CAgent
from keras.layers import Dense, Flatten, Input, concatenate
from keras.models import Sequential, Model
from keras.optimizers import rmsprop

config = dict(version=1,
              handlers={'console': {'class': 'logging.StreamHandler', 'level': 'INFO', 'formatter': 'main'},
                        'file': {'class': 'logging.FileHandler', 'level': 'INFO', 'filename': 'lunar_lander.log', 'formatter': 'main'}},
              formatters={'main': {'format': '[ %(levelname)s/%(processName)s] %(message)s'}},
              root={'level': 'DEBUG', 'handlers': ['console', 'file']},
              disable_existing_loggers=False)


import logging.config
logging.config.dictConfig(config)


def shape_reward(*args):
    s, a, r, s_prime, done = args
    def potential(state):
        x_pos = state[0]
        y_pos = state[1]

        return 10 * ((1 - abs(x_pos)) + (1 - y_pos)) # Encourage moving to 0,0 (center of landing pad)

    r = r + 0.99*potential(s_prime) - potential(s)
    return (s, a, r, s_prime, done)


if __name__ == '__main__':
    env = gym.make('LunarLander-v2')

    num_features = env.observation_space.shape[0]
    num_actions = env.action_space.n

    actor = Sequential([
        Dense(32, input_dim=num_features, activation='relu'),
        Dense(32, activation='relu'),
        Dense(units=num_actions, activation='softmax')
    ])
    actor.compile(loss='mse', optimizer=rmsprop(lr=0.0016))

    critic = Sequential([
        Dense(32, input_dim=num_features, activation='relu'),
        Dense(32, activation='relu'),
        Dense(units=1, activation='linear')
    ])
    critic.compile(loss='mse', optimizer=rmsprop(lr=0.0016))

    agent = A3CAgent(env=env, gamma=0.99, actor=actor, critic=critic, max_steps_per_episode=500, beta=2.5)
    agent.train(num_workers=3, max_episodes=500, train_every_n=5, render_every_n=0, preprocess_func=shape_reward)

    df = agent.history.get_episode_metrics()
    if df.shape[0] > 0:
        p = df.plot.line(x='episode_count', y='total_reward')
        p = df.plot.line(x='episode_count', y='mean_reward', ax=p)
        p.figure.show()
    pass