import gym
from deeprl.agents import ActorCriticAgent
from keras.layers import Dense, Input, concatenate
from keras.models import Sequential, Model
from keras.optimizers import rmsprop, sgd
from keras.regularizers import l2
from deeprl.memories import Memory
from deeprl.policies import NoisyPolicy

config = dict(version=1,
              handlers={'console': {'class': 'logging.StreamHandler', 'level': 'INFO'}},
              root={'level': 'DEBUG', 'handlers': ['console']},
              disable_existing_loggers=False)

import logging.config
logging.config.dictConfig(config)

env = gym.make('Pendulum-v0')

num_features = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]

actor = Sequential([
    Dense(32, input_dim=num_features, activation='relu'),
    Dense(32, activation='relu'),
    Dense(units=num_actions, activation='tanh')
])
actor.compile(loss='mse', optimizer=rmsprop(lr=1e-4, decay=0.000001, clipnorm=1.))

critic_state_input = Input(shape=(num_features,), name='CriticStateIn')
critic_action_input = Input(shape=(num_actions,), name='CriticActionIn')
critic_merged_input = concatenate([critic_state_input, critic_action_input])
critic_h1 = Dense(32, activation='relu', name='CriticH1', kernel_regularizer=l2(1e-2), bias_regularizer=l2(1e-2))(
    critic_merged_input)
critic_h2 = Dense(32, activation='relu', name='CriticH2', kernel_regularizer=l2(1e-2), bias_regularizer=l2(1e-2))(
    critic_h1)
critic_out = Dense(1, activation='linear', name='CriticOut', kernel_regularizer=l2(1e-2), bias_regularizer=l2(1e-2))(
    critic_h2)
critic = Model(inputs=[critic_state_input, critic_action_input], outputs=[critic_out])
critic.compile(sgd(lr=1e-3, clipnorm=5.), 'mse')

memory = Memory(maxlen=1e6, sample_size=32)
policy = NoisyPolicy(theta=0.15, sigma=0.15)

agent = ActorCriticAgent(env=env, actor=actor, critic=critic, policy=policy, memory=memory, max_steps_per_episode=500)
agent.train(max_episodes=1500, render_every_n=10, target_model_update=1e-3, frame_skip=1)