import keras.backend as K
import numpy as np

from .abstract import AbstractAgent


class ReinforceAgent(AbstractAgent):

    def __init__(self, model, **kwargs):
        super(ReinforceAgent, self).__init__(**kwargs)

        assert K.int_shape(model.input)[1] == self.num_features, \
            f"Model input dimensions of {K.int_shape(model.input)} do not match the observation space dimensions of {self.num_features}."

        assert K.int_shape(model.output)[1] == self.num_actions, \
            f"Model output dimensions of {K.int_shape(model.output)} do not match the action space dimensions of {self.num_actions}."

        self.model = model

        # TODO: Add trajectory memory?



    def choose_action(self, state):
        actions = self.model.predict_on_batch(state)

        if self.num_actions == 1:
            return actions
        else:
            assert np.sum(actions) == 1., 'Action probabilities must sum to one.'

            # Select the action to take
            return np.random.choice(np.arange(actions.size), p=actions.ravel())



    def _update_weights(self):
        # Compute gradient of action wrt weights
        # Scale by total return * discount factor
        pass


