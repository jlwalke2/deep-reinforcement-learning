import gym
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import rmsprop
from deeprl.agents import ReinforceAgent
from deeprl.memories import PrioritizedMemory

env = gym.make('Pendulum-v0')

num_features = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]

actor = Sequential([
    Dense(16, input_dim=num_features, activation='relu'),
    Dense(16, activation='relu'),
    Dense(units=num_actions, activation='sigmoid')
])
actor.compile(loss='mse', optimizer=rmsprop(lr=0.0016, decay=0.000001))

agent = ReinforceAgent(env=env, model=actor,
                       memory=PrioritizedMemory(maxlen=50000, sample_size=32),
                       max_steps_per_episode=500)

agent.train(max_episodes=1500, render_every_n=1)