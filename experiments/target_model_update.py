from deeprl.agents.DoubleDeepQAgent import DoubleDeepQAgent
from deeprl.memories import PrioritizedMemory
from deeprl.policies import EpsilonGreedyPolicy
from deeprl.utils.metrics import *
from keras.models import Sequential
from keras.layers import Dense
import gym

experiment_name = 'target_model_update'
num_agents = 10
num_episodes = 100



def train_fast_update(agent):
    agent.train(target_model_update=1e-2, max_episodes=num_episodes, render_every_n=None)

def train_medium_update(agent):
    agent.train(target_model_update=1e-4, max_episodes=num_episodes, render_every_n=None)


def train_slow_update(agent):
    agent.train(target_model_update=1e-6, max_episodes=num_episodes, render_every_n=None)


def build_agent(name):
    env = gym.make('LunarLander-v2')

    num_features = DoubleDeepQAgent._get_space_size(env.observation_space)
    num_actions = DoubleDeepQAgent._get_space_size(env.action_space)

    model = Sequential([
        Dense(64, input_dim=num_features, activation='relu'),
        Dense(64, activation='relu'),
        Dense(units=num_actions, activation='linear')
    ])
    model.compile(loss='mse', optimizer='sgd')

    agent = DoubleDeepQAgent(name=name, env=env, model=model,
                             policy=EpsilonGreedyPolicy(min=0.05, max=0.5, decay=0.999),
                             memory=PrioritizedMemory(maxlen=50000),
                             metrics=[EpisodeReturn(), RollingEpisodeReturn(), CumulativeReward(), EpisodeTime()],
                             gamma=0.99, max_steps_per_episode=500)

    return agent


if __name__ == '__main__':

    from experiments.experiment import Experiment

    agents = [(build_agent('FastUpdate'), num_agents, train_fast_update),
              (build_agent('MediumUpdate'), num_agents, train_medium_update),
              (build_agent('SlowUpdate'), num_agents, train_slow_update),
              ]

    e = Experiment(experiment_name, agents)

    e.run()

    import matplotlib.pyplot as plt
    p = e.get_plots(['CumulativeReward','EpisodeReward', 'RollingEpisodeReward50'])

    plt.show()

#     fig.savefig(experiment_name + '_avg_reward.png', dpi=500, linewidth=1)




