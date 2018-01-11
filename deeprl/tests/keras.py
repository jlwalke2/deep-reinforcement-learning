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

    def test_function(self):
        # Define loss tensor?
        # Define weight tensors?
        # Compute gradient of loss wrt weights
        # Name scopes?

        '''

        log(a | s) * (R - V)
        # Inputs:  states, action mask, returns, values
        # Outputs: loss?
        # Updates: gradient

        Compute objective function
        Compute gradient of weights wrt objective
        update weights
        '''
        model = Sequential([
            Dense(16, input_dim=4, activation='relu'),
            Dense(8, activation='relu'),
            Dense(units=2, activation='linear')
        ])
        model.compile(loss='mse', optimizer='sgd')
        orig_weights = model.get_weights()

        # Train model normally on a single input
        x = np.random.random((1,4))
        y_true = np.random.random((1,2))
        train_loss = model.train_on_batch(x, y_true)

        # Undo any weight updates made
        model.set_weights(orig_weights)

        # Build a Keras function that replicates the training function
        model = model.model
        input = model._feed_inputs + model._feed_targets + model._feed_sample_weights
        output = [model.total_loss]
        update = model.optimizer.get_updates(params=model._collected_trainable_weights, loss=model.total_loss)

        f = K.function(inputs=input,
                       outputs=output,
                       updates=update)

        # Run the function on a single input
        sample_weight = np.ones((y_true.shape[0],), dtype=K.floatx())
        f_loss = f([x, y_true, sample_weight])

        # Both methods should have accomplished the same thing
        self.assertEqual(train_loss, f_loss)

        # Create placeholder tensors for new arrays
        # Wrap loss function in closure
        # Define new train function

        def objective(y_pred, mask, advantage):
            y_pred = K.print_tensor(y_pred, 'Q(s,a) = ')
            mask = K.print_tensor(mask, 'Mask = ')
            advantage = K.print_tensor(advantage, 'Advantage = ')
            return K.print_tensor(-K.sum(K.log(y_pred) * mask, axis=-1) * advantage, 'loss=')

        mask = K.placeholder(shape=model.output.shape, name='mask')
        advantage = K.placeholder(ndim=1, name='advantage')
        loss = objective(model.output, mask, advantage)
        g = K.function(inputs=model._feed_inputs + [mask, advantage],
                       outputs=[loss],
                       updates=model.optimizer.get_updates(params=model._collected_trainable_weights, loss=loss))

        mask = np.array([0, 1.]).reshape((1,-1))
        advantage = np.array([2.])

        t1 = model.predict(x)
        t0 = g([x, mask, advantage])
        t2 = model.predict(x)

        pass


