import gym
from deeprl.agents.ddpg import ActorCriticAgent
from keras.layers import Dense, Input, concatenate
from keras.models import Sequential, Model
from keras.optimizers import rmsprop

from deeprl.memories import PrioritizedMemory
from deeprl.policies import EpsilonGreedyPolicy


env = gym.make('Pendulum-v0')

num_features = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]

actor = Sequential([
    Dense(16, input_dim=num_features, activation='relu'),
    Dense(16, activation='relu'),
    Dense(units=num_actions, activation='sigmoid')
])
actor.compile(loss='mse', optimizer=rmsprop(lr=0.0016, decay=0.000001))

state_in = Input(shape=(num_features,))
action_in = Input(shape=(1,))
critic = concatenate([state_in, action_in])
critic = Dense(16, activation='relu')(critic)
critic = Dense(16, activation='relu')(critic)
q_out = Dense(1, activation='linear')(critic)
critic = Model(inputs=[state_in, action_in], outputs=[q_out])
critic.compile(loss='mse', optimizer=rmsprop(lr=0.0016, decay=0.000001))


memory = PrioritizedMemory(maxlen=50000, sample_size=32)
policy = EpsilonGreedyPolicy(min=0.025, decay=0.96, exploration_episodes=1)

agent = ActorCriticAgent(env=env, actor=actor, critic=critic, policy=policy, memory=memory, max_steps_per_episode=500)

agent.train(max_episodes=1500, render_every_n=1, target_model_update=750)