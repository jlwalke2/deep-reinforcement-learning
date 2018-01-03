import unittest
import numpy as np
import keras.backend as K
from keras.layers import Dense
from keras.models import Sequential


class Test(unittest.TestCase):
    '''
    Not actual test cases
    Intended to confirm underlying Keras behavior.
    '''
    def test_gradients(self):
        '''
        Keras always tries to minimize the loss function and always *subtracts* the gradient.
        Therefore, instead of maximizing an objective function we must minimize the negative of the
        objective function.  Create a custom loss function that returns -1 * the objective.
        '''

        def positive_loss(y_actual, y_pred):
            # Standard MSE loss function
            return K.mean(K.square(y_pred - y_actual), axis=-1)

        def negative_loss(y_actual, y_pred):
            # Negative of standard MSE loss
            return -K.mean(K.square(y_pred - y_actual), axis=-1)

        dim_x = 4
        dim_y = 1
        x = np.random.random((1,4))
        y_actual = np.array([5])
        model_pos = Sequential([
            Dense(8, input_dim=dim_x, activation='relu'),
            Dense(8, activation='relu'),
            Dense(units=dim_y, activation='linear')
        ])
        model_neg = Sequential.from_config(model_pos.get_config())
        model_pos.compile('sgd', loss=positive_loss)
        model_neg.compile('sgd', loss=negative_loss)
        model_neg.set_weights(model_pos.get_weights())

        # Confirm that a positive loss results in loss being minimized.
        y_pred1 = model_pos.predict(x)
        error = model_pos.train_on_batch(x, y_actual)
        y_pred2 = model_pos.predict(x)
        self.assertLess(abs(y_actual - y_pred2), abs(y_actual - y_pred1))

        # Confirm that a negative loss results in loss being maximized
        error = model_neg.train_on_batch(x, y_actual)
        y_pred3 = model_neg.predict(x)
        self.assertGreater(abs(y_actual - y_pred3), abs(y_actual - y_pred1))


