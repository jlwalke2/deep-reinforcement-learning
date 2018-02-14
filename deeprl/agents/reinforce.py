import logging
import keras.backend as K
import numpy as np
import warnings

from .abstract import AbstractAgent
from ..memories import TrajectoryMemory
from ..utils.misc import unwrap_model


logger = logging.getLogger(__name__)

class ReinforceAgent(AbstractAgent):

    def __init__(self, model, **kwargs):
        # Instantiate a Memory if not provided
        if 'memory' in kwargs:
            warnings.warn('If the "memory" parameter is set it is critical that the memory return samples in order.')
        else:
            kwargs['memory'] = TrajectoryMemory()

        super(ReinforceAgent, self).__init__(**kwargs)

        # Validate the model's input/output dimensions
        # Do this after calling parent constructor so that attributes are set.
        assert K.int_shape(model.input)[1] == self.num_features, \
            f"Model input dimensions of {K.int_shape(model.input)} do not match the observation space dimensions of {self.num_features}."

        assert K.int_shape(model.output)[1] == self.num_actions, \
            f"Model output dimensions of {K.int_shape(model.output)} do not match the action space dimensions of {self.num_actions}."


        self.model = model
        model = unwrap_model(model)

        # Define a custom objective function to maximize
        def objective(action, mask, total_return):
            # Since Keras optimizers always minimize "loss" we'll have to minimize the negative of the objective

            # Gradient of log(pi(s))
            return -K.sum(K.log(action + 1e-30) * mask * total_return, axis=-1)

        mask = K.placeholder(shape=model.output.shape, name='mask')
        advantage = K.placeholder(shape=(None, 1), name='advantage')
        loss = objective(model.output, mask, advantage)  # Compute objective/loss tensor

        # Build a Keras function to run in the inputs through the model, return the outputs, and perform the
        # (weight) updates created by the optimizer
        self._train_function = K.function(inputs=model._feed_inputs + [mask, advantage],
                              outputs=[loss],
                              updates=model.optimizer.get_updates(params=model._collected_trainable_weights, loss=loss, constraints={}))

        # Construct an upper triangular matrix where each diagonal 0..k = the discount factor raised to that power
        # Once constructed, the n-step return can be computed as Gr where G is the matrix of discount factors
        # and r is the vector of rewards observed at each time step
        # We compute this ahead of time since it's static and pow() computations are slow, even in numpy
        self._discounts = sum([np.diagflat(np.ones(self.max_steps_per_episode - i) * self.gamma ** i, k=i) for i in range(self.max_steps_per_episode)])




    def choose_action(self, state):
        actions = self.model.predict_on_batch(state)

        logging.debug(actions)

        if self.num_actions == 1:
            return actions
        else:
            # Select the action to take
            return np.random.choice(np.arange(actions.size), p=actions.ravel())



    def _update_weights(self):
        # Only train when a full trajectory is observed
        if not self._status.episode_done:
            return

        states, actions, rewards, s_primes, flags = self.memory.sample()
        batch_size = states.shape[0]

        # Mask out action values for everything except that action taken from s
        if self.num_actions > 1:
            mask = np.zeros((batch_size, self.num_actions))
            mask[range(batch_size), actions.astype('int32').ravel()] = 1
        else:
            mask = np.ones((batch_size, 1))

        # If episode terminated early then the number of states < max steps per episode
        # Select a submatrix from _discounts that matches the number of steps observed.
        indx = min(rewards.size, self._discounts.shape[0])
        discount = self._discounts[:indx, :indx]
        # Total discounted return from s
        R = np.dot(discount, rewards)

        # Compute gradient of action wrt weights
        # Scale by total return * discount factor
        loss = self._train_function([states, mask, R])

        pass


