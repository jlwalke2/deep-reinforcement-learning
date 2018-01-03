import unittest
from keras.layers import Dense
from keras.models import Sequential
from ..utils.async import ModelManager

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
            pass

        finally:
            manager.shutdown()

