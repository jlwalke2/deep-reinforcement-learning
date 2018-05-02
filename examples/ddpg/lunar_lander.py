import gym
from deeprl.agents.ddpg import ActorCriticAgent
import keras.backend as K
from keras.layers import Dense, Input, concatenate
from keras.models import Sequential, Model
from keras.optimizers import rmsprop, sgd
from deeprl.memories import Memory, PrioritizedMemory
from deeprl.utils.metrics import InitialStateValue
from deeprl.policies import NoisyPolicy

import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
if len(logger.handlers) > 0:
    logger.handlers[0].setLevel(logging.INFO)

# fileHandler = logging.FileHandler('logs/lunarlander_ac.log')
# fileHandler.setLevel(logging.DEBUG)
# logger.addHandler(fileHandler)

env = gym.make('LunarLanderContinuous-v2')

num_features = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]

actor = Sequential([
    Dense(128, input_dim=num_features, activation='relu'),
    Dense(128, activation='relu'),
    Dense(units=num_actions, activation='tanh')
])
actor.compile(loss='mse', optimizer=rmsprop(lr=1e-4)) #optimizer=sgd(lr=1e-13))

critic_state_input = Input(shape=(num_features,), name='critic_state_input')
critic_action_input = Input(shape=(num_actions,), name='critic_action_input')
critic_merged_input = concatenate([critic_state_input, critic_action_input])
critic_h1 = Dense(128, activation='relu', name='critic_h1')(critic_merged_input)
critic_h2 = Dense(128, activation='relu', name='critic_h2')(critic_h1)
critic_out = Dense(1, activation='linear', name='CriticOut')(critic_h2)
critic = Model(inputs=[critic_state_input, critic_action_input], outputs=[critic_out])
critic.compile(sgd(lr=1e-3, clipnorm=5.), 'mse')

memory = PrioritizedMemory(maxlen=1e6, sample_size=32)
agent = ActorCriticAgent(env=env, actor=actor, critic=critic, memory=memory,
                         policy=NoisyPolicy(0.15, 0.5, clip=env.action_space),
                         max_steps_per_episode=500,
                         tb_path = 'tensorboard')

agent.train(max_episodes=10000, render_every_n=50, target_model_update=1e-4)
