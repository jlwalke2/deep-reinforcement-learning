import random as rnd
import numpy as np
import gym
from Agents import DoubleDeepQAgent
from Memories import Memory
from Policies import BoltzmannPolicy

import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.optimizers import sgd, rmsprop, adam, adadelta

SEED = 123

env = gym.make('CartPole-v0')
env.seed(SEED)

num_features = env.observation_space.shape[0]
num_actions = env.action_space.n

model = Sequential()
model.add(Dense(16, input_dim=num_features, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(units=num_actions, activation='linear'))
model.compile(loss='mse', optimizer=rmsprop(lr=1e-3))
print(model.summary())

memory = Memory(50000)
policy = BoltzmannPolicy()

agent = DoubleDeepQAgent(env, model, policy, memory, gamma=0.99, api_key='sk_giCGTLHbRVjTTS7YYMtuA', seed=SEED)

agent.train(target_model_update=1e-2, upload=False, max_episodes=500)