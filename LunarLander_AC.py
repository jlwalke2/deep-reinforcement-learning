import gym
from deeprl.agents.ActorCriticAgent import ActorCriticAgent
import keras.backend as K
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import rmsprop

from deeprl.memories import PrioritizedMemory

env = gym.make('LunarLander-v2')

num_features = env.observation_space.shape[0]
num_actions = env.action_space.n

def actor_loss(y_true, y_pred):
    return K.mean(y_true - y_pred, axis=-1)

actor = Sequential([
    Dense(16, input_dim=num_features, activation='relu'),
    Dense(16, activation='relu'),
    Dense(units=num_actions, activation='softmax')
])
actor.compile(loss=actor_loss, optimizer=rmsprop(lr=0.0016, decay=0.000001))


critic = Sequential([
    Dense(16, input_dim=num_features, activation='relu'),
    Dense(16, activation='relu'),
    Dense(units=1, activation='linear')
])
critic.compile(loss='mse', optimizer=rmsprop(lr=0.0016, decay=0.000001, clipnorm=0.001))


memory = PrioritizedMemory(50000, sample_size=32)
#policy = EpsilonGreedyPolicy(min=0.025, decay=0.96, exploration_episodes=1)

agent = ActorCriticAgent(env=env, actor=actor, critic=critic, memory=memory, max_steps_per_episode=500)
import logging
agent.logger.setLevel(logging.DEBUG)
agent.train(max_episodes=1000, render_every_n=25, target_model_update=750)

df = agent.logger.get_episode_metrics()
p = df.plot.line(x='episode_count', y='total_reward')
p = df.plot.line(x='episode_count', y='mean_reward', ax=p)
p.figure.show()
pass