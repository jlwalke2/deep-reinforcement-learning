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


if __name__ == '__main__':
    env = gym.make('BipedalWalkerHardcore-v2')

    # Continuous action and observation spaces
    num_features = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]

    actor_input = Input(shape=(num_features,), name='ActorIn')
    actor_h1 = Dense(32, activation='relu', name='ActorH1')(actor_input)
    actor_h2 = Dense(32, activation='relu', name='ActorH2')(actor_h1)
    actor_out = concatenate([Dense(1, activation='sigmoid')(actor_h2) for i in range(num_actions)])
    actor = Model(inputs=[actor_input], outputs=[actor_out])
    actor.compile(loss='mse', optimizer=rmsprop(lr=0.0016))

    # TODO: Method to validate network input & output against env spaces

    critic = Sequential([
        Dense(32, input_dim=num_features, activation='relu'),
        Dense(32, activation='relu'),
        Dense(units=1, activation='linear')
    ])
    critic.compile(loss='mse', optimizer=rmsprop(lr=0.0016))

    agent = A3CAgent(env=env, gamma=0.99, actor=actor, critic=critic, max_steps_per_episode=500, beta=2.5)
    agent.train(num_workers=3, max_episodes=500, train_every_n=5, render_every_n=1, frame_skip=4)

    df = agent.history.get_episode_metrics()
    if df.shape[0] > 0:
        p = df.plot.line(x='episode_count', y='total_reward')
        p = df.plot.line(x='episode_count', y='mean_reward', ax=p)
        p.figure.show()
    pass