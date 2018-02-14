import gym
import logging
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import rmsprop
from deeprl.agents import ReinforceAgent


logger = logging.getLogger()
if not logger.isEnabledFor(logging.INFO):
    logger.handlers.clear()
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    #logger.setLevel(logging.DEBUG)
    logger.setLevel(logging.INFO)

env = gym.make('Pendulum-v0')

num_features = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]

actor = Sequential([
    Dense(16, input_dim=num_features, activation='relu'),
    Dense(16, activation='relu'),
    Dense(units=num_actions, activation='linear')
])
actor.compile(loss='mse', optimizer=rmsprop(lr=0.0016, decay=0.000001))

agent = ReinforceAgent(env=env, model=actor, max_steps_per_episode=50)
agent.train(max_episodes=5000, render_every_n=100)