import gym
from deeprl.agents import DoubleDeepQAgent
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import rmsprop
from deeprl.memories import PrioritizedMemory, Memory
from deeprl.policies import EpsilonGreedyPolicy, BoltzmannPolicy
from deeprl.utils import set_seed, animated_plot
import deeprl.utils.metrics as metrics

import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.handlers[0].setLevel(logging.INFO)
#set_seed(0)


env = gym.make('LunarLander-v2')

num_features = env.observation_space.shape[0]
num_actions = env.action_space.n

model = Sequential([
    Dense(64, input_dim=num_features, activation='relu'),
    Dense(64, activation='relu'),
    Dense(units=num_actions, activation='linear')
])
model.compile(loss='mse', optimizer=rmsprop(lr=0.0016, decay=0.000001, clipnorm=1.))

#memory = Memory(maxlen=250000, sample_size=32)
memory = PrioritizedMemory(maxlen=50000, sample_size=32)
policy = BoltzmannPolicy()
#policy = EpsilonGreedyPolicy(min=0.1, decay=0.99, exploration_episodes=1)

agent = DoubleDeepQAgent(env=env, model=model, policy=policy, memory=memory, gamma=0.99,
                         max_steps_per_episode=500, tb_path='tensorboard')

agent.train(target_model_update=1e-3, max_episodes=100, render_every_n=0)
agent.test(num_episodes=10, render_every_n=1)
