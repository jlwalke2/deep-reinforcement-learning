
import gym
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import rmsprop

from deeprl.agents import DoubleDeepQAgent
from deeprl.memories import PrioritizedMemory
from deeprl.policies import BoltzmannPolicy
from deeprl.utils import animated_plot, set_seed
from deeprl.utils.metrics import *


config = dict(version=1,
              handlers={'console': {'class': 'logging.StreamHandler', 'level': 'INFO'}},
              root={'level': 'DEBUG', 'handlers': ['console']},
              disable_existing_loggers=False)

import logging.config
logging.config.dictConfig(config)

set_seed(0)
env = gym.make('CartPole-v0')

num_features = env.observation_space.shape[0]
num_actions = env.action_space.n

model = Sequential()
model.add(Dense(16, input_dim=num_features, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(units=num_actions, activation='linear'))
model.compile(loss='mse', optimizer=rmsprop(lr=1e-3))
print(model.summary())

memory = PrioritizedMemory(maxlen=50000)
from deeprl.memories import Memory
memory = Memory(maxlen=50000)
policy = BoltzmannPolicy()


agent = DoubleDeepQAgent(env=env, model=model, policy=policy, memory=memory, gamma=0.99,
                         metrics=[EpisodeReturn(), RollingEpisodeReturn(), CumulativeReward(), EpisodeTime()],
                         max_steps_per_episode=300)

# plt, anim = animated_plot(agent.history.get_episode_metrics, ['EpisodeReward', 'RollingEpisodeReward50'])
# plt.show(block=False)

agent.train(target_model_update=1e-2, upload=False, max_episodes=500, render_every_n=0)

# df = agent.history.get_episode_metrics()
# df.to_csv('cartpole.csv')
# p = df.plot.line(y='EpisodeReward')
# p = df.plot.line(y='RollingEpisodeReward50', ax=p)
# p.figure.show()
pass