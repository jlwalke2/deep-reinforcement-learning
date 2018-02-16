import gym
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import adam

from deeprl.agents import DoubleDeepQAgent
from deeprl.memories import PrioritizedMemory
from deeprl.policies import EpsilonGreedyPolicy
from deeprl.utils import animated_plot

env = gym.make('MountainCar-v0')

num_features = env.observation_space.shape[0]
num_actions = env.action_space.n

model = Sequential()
model.add(Dense(16, input_dim=num_features, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(units=num_actions, activation='linear'))
model.compile(loss='mse', optimizer=adam())
print(model.summary())

def shape_reward(*args):
    # state is (position, velocity)
    s, a, r, s_prime, done = args
    def potential(state):
        return 1. if np.all(state > 0) or np.all(state < 0) else 0

    r = r + 0.99*potential(s_prime) - potential(s)

    return args


memory = PrioritizedMemory(maxlen=50000)
policy = EpsilonGreedyPolicy(min=0.05, max=0.5, decay=0.999)
#policy = BoltzmannPolicy()
agent = DoubleDeepQAgent(env=env, model=model, policy=policy, memory=memory, gamma=0.99, max_steps_per_episode=1000)
agent.preprocess_state = shape_reward

plt, anim = animated_plot(agent.history.get_episode_metrics, ['total_reward', 'avg_reward'])
plt.show(block=False)

agent.train(target_model_update=1e-2, upload=False, max_episodes=1000, render_every_n=10)

df = agent.history.get_episode_metrics()
p = df.plot.line(x='episode_count', y='total_reward')
p = df.plot.line(x='episode_count', y='mean_reward', ax=p)
p.figure.show()
pass