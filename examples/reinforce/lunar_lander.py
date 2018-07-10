import gym
import logging
from deeprl.agents import ReinforceAgent
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import rmsprop
from deeprl.memories import PrioritizedMemory

logger = logging.getLogger()
if not logger.isEnabledFor(logging.INFO):
    logger.handlers.clear()
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    #logger.setLevel(logging.DEBUG)
    logger.setLevel(logging.INFO)

env = gym.make('LunarLander-v2')
num_features = env.observation_space.shape[0]
num_actions = env.action_space.n

model = Sequential([
    Dense(64, input_dim=num_features, activation='relu'),
    Dense(64, activation='relu'),
    Dense(units=num_actions, activation='softmax')
])
model.compile(loss='mse', optimizer=rmsprop(lr=0.0016, decay=0.000001, clipnorm=1.))

memory = PrioritizedMemory(maxlen=50000, sample_size=32)
agent = ReinforceAgent(env=env, model=model, max_steps_per_episode=500)
agent.train(max_episodes=10000, render_every_n=50)

