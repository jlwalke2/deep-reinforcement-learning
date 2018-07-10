from deeprl.agents.DoubleDeepQAgent import DoubleDeepQAgent
from deeprl.memories import PrioritizedMemory
from deeprl.policies import EpsilonGreedyPolicy
from deeprl.utils.metrics import *
from keras.models import Sequential
from keras.layers import Dense
import gym

env = gym.make('LunarLander-v2')

num_features = DoubleDeepQAgent._get_space_size(env.observation_space)
num_actions = DoubleDeepQAgent._get_space_size(env.action_space)

model = Sequential([
    Dense(64, input_dim=num_features, activation='relu'),
    Dense(64, activation='relu'),
    Dense(units=num_actions, activation='linear')
])
model.compile(loss='mse', optimizer='sgd')

import deeprl.agents.AbstractAgent
import logging, logging.handlers

root = logging.getLogger()
root.handlers[0].setLevel(logging.INFO)

file = logging.FileHandler('test_nan.log', mode='w')
file.setLevel(logging.DEBUG)
deeprl.agents.AbstractAgent.logger.addHandler(file)
deeprl.agents.AbstractAgent.logger.setLevel(logging.DEBUG)


agent = DoubleDeepQAgent(env=env, model=model,
                         policy=EpsilonGreedyPolicy(min=0.05, max=0.5, decay=0.999),
                         memory=PrioritizedMemory(maxlen=50000),
                         metrics=[EpisodeReturn(), RollingEpisodeReturn(), CumulativeReward(), EpisodeTime()],
                         gamma=0.99, max_steps_per_episode=500)

agent.train(target_model_update=1e-4, max_episodes=1000, render_every_n=None)