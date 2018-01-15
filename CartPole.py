import gym
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import rmsprop

from deeprl.agents.DoubleDeepQAgent import DoubleDeepQAgent
from deeprl.memories import PrioritizedMemory
from deeprl.policies import BoltzmannPolicy
from deeprl.utils import animated_plot, set_seed

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
policy = BoltzmannPolicy()


agent = DoubleDeepQAgent(env=env, model=model, policy=policy, memory=memory, gamma=0.99, max_steps_per_episode=500)

plt, anim = animated_plot(agent.metrics.get_episode_metrics, ['total_reward','avg_reward'])
plt.show(block=False)

agent.train(target_model_update=1e-2, upload=False, max_episodes=100, render_every_n=1001)

df = agent.metrics.get_episode_metrics()
df.to_csv('cartpole.csv')
p = df.plot.line(y='total_reward')
p = df.plot.line(y='avg_reward', ax=p)
p.figure.show()
pass