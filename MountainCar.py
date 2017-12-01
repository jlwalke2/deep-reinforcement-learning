import gym
from Agents import DoubleDeepQAgent
from Memories import Memory
from Policies import EpsilonGreedyPolicy

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import rmsprop

SEED = 123

env = gym.make('MountainCar-v0')

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
policy = EpsilonGreedyPolicy(min=0.05, decay=0.99)

agent = DoubleDeepQAgent(env, model, policy, memory, gamma=0.99, seed=SEED)

agent.train(target_model_update=1e-2, upload=False, max_episodes=200)