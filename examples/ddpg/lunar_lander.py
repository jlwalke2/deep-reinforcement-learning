import gym
from deeprl.agents.ddpg import ActorCriticAgent
import keras.backend as K
from keras.layers import Dense, Input
from keras.models import Sequential, Model
from keras.optimizers import rmsprop, sgd
from deeprl.memories import PrioritizedMemory
from deeprl.utils.metrics import InitialStateValue

import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
if len(logger.handlers) > 0:
    logger.handlers[0].setLevel(logging.INFO)

fileHandler = logging.FileHandler('logs/lunarlander_ac.log')
fileHandler.setLevel(logging.DEBUG)
logger.addHandler(fileHandler)

env = gym.make('LunarLander-v2')

num_features = env.observation_space.shape[0]
num_actions = env.action_space.n

def actor_loss(y_true, y_pred):
    return K.mean(y_true - y_pred, axis=-1)



#
# shared_1 = Dense(64, activation='relu', name='Shared1')
# shared_2 = Dense(32, activation='relu', name='Shared2')
#
# actor_in = Input(shape=(num_features,), name='ActorIn')
# actor_h_1 = shared_1(actor_in)
# actor_h_2 = shared_2(actor_h_1)
# actor_out = Dense(units=num_actions, activation='softmax', name='ActorOut')(actor_h_2)
# actor = Model(inputs=[actor_in], outputs=[actor_out])
# actor.compile(loss=actor_loss, optimizer=rmsprop(lr=0.0016, decay=0.000001, clipnorm=0.5))
#
# critic_in = Input(shape=(num_features,), name='CriticIn')
# critic_h_1 = shared_1(critic_in)
# critic_h_2 = shared_2(critic_h_1)
# critic_out = Dense(units=1, activation='linear', name='CriticOut')(critic_h_2)
# critic = Model(inputs=[critic_in], outputs=[critic_out])
# critic.compile(loss='mse', optimizer=rmsprop(lr=0.0016, decay=0.000001, clipnorm=0.001))

actor = Sequential([
    Dense(16, input_dim=num_features, activation='relu'),
    Dense(16, activation='relu'),
    Dense(units=num_actions, activation='softmax')
])
actor.compile(loss=actor_loss, optimizer=rmsprop(lr=0.0016, decay=0.000001)) #optimizer=sgd(lr=1e-13))

critic = Sequential([
    Dense(16, input_dim=num_features, activation='relu'),
    Dense(16, activation='relu'),
    Dense(units=1, activation='linear')
])
critic.compile(loss='mse', optimizer=rmsprop(lr=0.0016, decay=0.000001))  #sgd(lr=1e-13))


memory = PrioritizedMemory(maxlen=50000, sample_size=32)
agent = ActorCriticAgent(env=env, actor=actor, critic=critic, memory=memory, max_steps_per_episode=500)
agent.wire_events(InitialStateValue(env, critic))

agent.train(max_episodes=10000, render_every_n=50, target_model_update=1e-4)

# df = agent.logger.get_episode_metrics()
# p = df.plot.line(x='episode_count', y='total_reward')
# p = df.plot.line(x='episode_count', y='mean_reward', ax=p)
#p.figure.show()
pass