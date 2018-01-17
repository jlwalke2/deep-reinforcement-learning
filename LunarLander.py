import gym
from deeprl.agents import DoubleDeepQAgent
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import rmsprop
from deeprl.memories import PrioritizedMemory, Memory
from deeprl.policies import EpsilonGreedyPolicy, BoltzmannPolicy
from deeprl.utils import set_seed, animated_plot

#set_seed(0)


env = gym.make('LunarLander-v2')

num_features = env.observation_space.shape[0]
num_actions = env.action_space.n

model = Sequential([
    Dense(64, input_dim=num_features, activation='relu'),
    Dense(64, activation='relu'),
    Dense(units=num_actions, activation='linear')
])
model.compile(loss='mse', optimizer=rmsprop(lr=0.0016, decay=0.000001))

def shape_reward(*args):
    s, a, r, s_prime, done = args
    def potential(state):
        x_pos = state[0]
        y_pos = state[1]

        return (1 - abs(x_pos)) + (1 - y_pos) # Encourage moving to 0,0 (center of landing pad)

    r = r + 0.99*potential(s_prime) - potential(s)

    return (s, a, r, s_prime, done)

#memory = Memory(maxlen=250000, sample_size=32)
memory = PrioritizedMemory(maxlen=50000, sample_size=32)
policy = BoltzmannPolicy()
#policy = EpsilonGreedyPolicy(min=0.1, decay=0.99, exploration_episodes=1)

agent = DoubleDeepQAgent(env=env, model=model, policy=policy, memory=memory, gamma=0.99, max_steps_per_episode=500)
#agent.preprocess_state = shape_reward

plt, anim = animated_plot(agent.history.get_episode_metrics, ['total_reward', 'avg_reward'])
plt.show(block=False)
agent.train(target_model_update=1e-3, max_episodes=1000, render_every_n=50)

df = agent.history.get_episode_metrics()
df.to_csv('lunar_lander.csv')
p = df['total_reward'].plot()
p = df['avg_reward'].plot(ax=p)
p.figure.show()
pass

