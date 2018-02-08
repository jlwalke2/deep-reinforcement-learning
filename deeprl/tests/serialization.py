import unittest
from ..utils.async import ModelManager
import gym
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import rmsprop

from deeprl.agents.DoubleDeepQAgent import DoubleDeepQAgent
from deeprl.memories import PrioritizedMemory
from deeprl.policies import BoltzmannPolicy, EpsilonGreedyPolicy
from deeprl.utils.metrics import *
import multiprocessing


class Test(unittest.TestCase):
    def test_shared_model_weights(self):
        import numpy as np

        try:
            manager = ModelManager()
            manager.start()

            initial_w1 = [np.random.random((3,3)) for _ in range(5)]
            model1 = manager.Model(None, None, None, None, None, initial_w1)

            initial_w2 = [np.random.random((4, 4)) for _ in range(5)]
            model2 = manager.Model(None, None, None, None, initial_w2)

            def assert_equal(weights1, weights2):
                for w1, w2 in zip(weights1, weights2):
                    np.testing.assert_array_equal(w1, w2)

            assert_equal(initial_w1, model1.get_weights())
            assert_equal(initial_w2, model2.get_weights())
        finally:
            manager.shutdown()

    def test_shared_model_optimizer(self):
        model = Sequential([
            Dense(16, input_dim=4, activation='relu'),
            Dense(16, activation='relu'),
            Dense(units=2, activation='softmax')
        ])
        model.compile('sgd', 'mse')

        try:
            manager = ModelManager()
            manager.start()

            model1 = manager.Model(model.get_config(), type(model.optimizer), model.optimizer.get_config(), model.get_weights())
            t0 = model1.get_optimizer()

        finally:
            manager.shutdown()

    def test_shared_monitor(self):
        try:
            manager = ModelManager()
            manager.start()

            metrics = manager.Monitor()
            metrics.info('Testing...')


        finally:
            manager.shutdown()

    def test_doubleqagent(self):


        env = gym.make('CartPole-v0')

        num_features = env.observation_space.shape[0]
        num_actions = env.action_space.n

        model = Sequential()
        model.add(Dense(16, input_dim=num_features, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(units=num_actions, activation='linear'))
        model.compile(loss='mse', optimizer=rmsprop(lr=1e-3))

        agent = DoubleDeepQAgent(env=env, model=model,
                                 policy=BoltzmannPolicy(),
                                 memory=PrioritizedMemory(maxlen=50000),
                                 metrics=[EpisodeReturn(), RollingEpisodeReturn(), CumulativeReward(), EpisodeTime()],
                                 gamma=0.99, max_steps_per_episode=500)

        import pickle

        s = agent.__getstate__()
        t0 = pickle.dumps(agent)
        t1 = pickle.loads(t0)

        agent = DoubleDeepQAgent(env=env, model=model,
                                 policy=EpsilonGreedyPolicy(min=0.05, max=0.5, decay=0.999),
                                 memory=PrioritizedMemory(maxlen=50000),
                                 metrics=[EpisodeReturn(), RollingEpisodeReturn(), CumulativeReward(), EpisodeTime()],
                                 gamma=0.99, max_steps_per_episode=1000)

        s = agent.__getstate__()
        t0 = pickle.dumps(agent)
        t1 = pickle.loads(t0)


        pass

