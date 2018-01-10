from deeprl.agents.DoubleDeepQAgent import DoubleDeepQAgent
from deeprl.memories import PrioritizedMemory
from deeprl.policies import EpsilonGreedyPolicy
from deeprl.utils.async import ModelManager
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import rmsprop
import gym
import multiprocessing

def run_agent_no_shaping(name, metrics):
    env = gym.make('LunarLander-v2')

    num_features = DoubleDeepQAgent._get_space_size(env.observation_space)
    num_actions = DoubleDeepQAgent._get_space_size(env.action_space)

    model = Sequential([
        Dense(64, input_dim=num_features, activation='relu'),
        Dense(64, activation='relu'),
        Dense(units=num_actions, activation='linear')
    ])
    model.compile(loss='mse', optimizer=rmsprop(lr=0.0016, decay=0.000001))

    agent = DoubleDeepQAgent(name=name, env=env, model=model,
                             policy=EpsilonGreedyPolicy(min=0.025, decay=0.96, exploration_episodes=1),
                             memory=PrioritizedMemory(maxlen=50000, sample_size=32),
                             logger=multiprocessing.log_to_stderr(), metrics=metrics, gamma=0.99, max_steps_per_episode=500)

    agent.train(target_model_update=750, max_episodes=10, render_every_n=10)


def run_agent_with_shaping(name, metrics):
    env = gym.make('LunarLander-v2')

    num_features = DoubleDeepQAgent._get_space_size(env.observation_space)
    num_actions = DoubleDeepQAgent._get_space_size(env.action_space)

    model = Sequential([
        Dense(64, input_dim=num_features, activation='relu'),
        Dense(64, activation='relu'),
        Dense(units=num_actions, activation='linear')
    ])
    model.compile(loss='mse', optimizer=rmsprop(lr=0.0016, decay=0.000001))

    agent = DoubleDeepQAgent(name=name, env=env, model=model,
                             policy=EpsilonGreedyPolicy(min=0.025, decay=0.96, exploration_episodes=1),
                             memory=PrioritizedMemory(maxlen=50000, sample_size=32),
                             logger=multiprocessing.log_to_stderr(), metrics=metrics, gamma=0.99, max_steps_per_episode=500)

    agent.train(target_model_update=750, max_episodes=10, render_every_n=10)


if __name__ == '__main__':
    manager = ModelManager()
    manager.start()
    metrics = manager.Monitor()
    logger = multiprocessing.log_to_stderr()

    processes = [multiprocessing.Process(target=run_agent_no_shaping, args=('AgentNoShape', metrics), name='Worker 1'),
                 multiprocessing.Process(target=run_agent_with_shaping, args=('AgentWithShape', metrics), name='Worker 2')]

    for p in processes:
        p.start()

    for p in processes:
        p.join()

    df = metrics.get_episode_metrics()
    print(df)
    pass


# multiple agents
# multiple run funcs
# multiprocessing

