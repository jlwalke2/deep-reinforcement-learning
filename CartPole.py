import gym
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import rmsprop

from deeprl.agents.DoubleDeepQAgent import DoubleDeepQAgent
from Memories import PrioritizedMemory
from policies import BoltzmannPolicy

SEED = 123

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
policy = BoltzmannPolicy()


agent = DoubleDeepQAgent(env=env, model=model, policy=policy, memory=memory, gamma=0.99, max_steps_per_episode=500, seed=SEED)

#import logging
#agent.policy.logger.setLevel(logging.DEBUG)
agent.train(target_model_update=1e-2, upload=False, max_episodes=200)

df = agent.logger.get_episode_metrics()
p = df.plot.line(x='episode_count', y='total_reward')
p = df.plot.line(x='episode_count', y='mean_reward', ax=p)
p.figure.show()
pass