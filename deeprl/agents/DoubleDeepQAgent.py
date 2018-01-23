import gym
import numpy as np
from ..utils.misc import keras2dict, dict2keras
from .DeepQAgent import DeepQAgent


class DoubleDeepQAgent(DeepQAgent):
    def __init__(self, **kwargs):
        super(DoubleDeepQAgent, self).__init__(**kwargs)

        # Create the target model.  Same architecture as DeepQ agent, but separate weights
        self.target_model = self._clone_model(self.model)

    def choose_action(self, state):
        q_values = self.model.predict_on_batch(state)

        assert np.any(np.isnan(q_values)) == False, 'Q-Values may not be NaN: {}'.format(q_values)

        return self.policy(q_values)

    def train(self, target_model_update: float, **kwargs):
        if target_model_update <= 0:
            raise ValueError('target_model_update must be a positive number.')

        self._target_model_update = target_model_update

        super(DoubleDeepQAgent, self).train(**kwargs)


    def _update_target_model(self, ratio=None):
        if ratio:
            # Soft model update
            w_t = self.target_model.get_weights()
            w_o = self.model.get_weights()

            for layer in range(len(w_o)):
                w_t[layer] = ratio * w_o[layer] + (1.0 - ratio) * w_t[layer]
            self.target_model.set_weights(w_t)
        else:
            # Hard update
            self.target_model.set_weights(self.model.get_weights())

    def _update_weights(self, n=0):
        '''Randomly select experiences from the replay buffer to train on.'''
        if not n:
            n = self.memory.sample_size

        states, actions, rewards, s_primes, flags = self.memory.sample(n)

        assert states.shape == (n, self.num_features)
        assert s_primes.shape == (n, self.num_features)
        assert actions.shape == (n, 1)
        assert rewards.shape == (n, 1)
        assert flags.shape == (n, 1)

        self.train_start(num_samples=n, s=states, a=actions, r=rewards, s_prime=s_primes, episode_done=flags)

        # Double Deep Q-Learning
        # Use online model to pick best action from s'
        s_prime_actions = self.model.predict_on_batch(s_primes)
        best_actions = np.argmax(s_prime_actions, axis=1)

        # But use target model to determine Q value of selected action from s'
        s_prime_values = self.target_model.predict_on_batch(s_primes)
        best_values = s_prime_values[range(n), best_actions]

        # If s' is a terminal state then it has no additional value, so zero out its q-value
        best_values[flags.ravel()] = 0

        # Compute the new Q value of s
        updated_targets = rewards + self.gamma * best_values.reshape(rewards.shape)

        # Target and estimated q-values should be same for any action not taken from s
        # This ensures the difference between targte & estimated q-values is 0 and only the selected action influences
        # training.
        estimate = self.model.predict_on_batch(states)
        targets = estimate.copy()
        targets[range(n), actions.astype('int32').ravel()] = updated_targets[:, 0] + 1e-10 # Add small epsilon to avoid 0 error

        #self.calculate_error(target=targets, estimate=estimate)

        diff = targets[range(n), actions.astype('int32').ravel()] - updated_targets[:, 0]
        # targets[range(n), actions.astype('int32').ravel()] = updated_targets[:, 0]

        error = self.model.train_on_batch(states, targets)

        self.train_end(num_samples=n, s=states, a=actions, r=rewards, s_prime=s_primes, episode_done=flags,
                       target=targets, delta=diff, error=error)

        return error

    def _raise_step_end_event(self, **kwargs):
        assert self._status.episode is not None
        assert 'total_steps' in self._status

        # Hard update of target model weights every N steps
        if self._status.total_steps % self._target_model_update == 0:
            self._update_target_model()
        elif self._target_model_update < 1:
            # Soft weight update after every step
            self._update_target_model(self._target_model_update)

        # Train model
        if self._status.total_steps > self._steps_before_training:
            error = self._update_weights()

        super(DoubleDeepQAgent, self)._raise_step_end_event(**kwargs)


    def __getstate__(self):
        state = self.__dict__.copy()

        # Convert Keras models into dictionaries of serializable objects
        state['model'] = keras2dict(self.model)
        state['target_model'] = keras2dict(self.target_model)
        state['env'] = self.env.spec.id

        # TODO: handle objects with loggers
        for key in ['logger']:
            if key in state:
                del state[key]

        return state

    def __setstate__(self, state):
        # Rebuild the Keras models
        self.model = dict2keras(state['model'])
        self.target_model = dict2keras(state['target_model'])
        self.env = gym.make(state['env'])

        import logging
        #state['logger'] = logging.getLogger()
        del state['model']
        del state['target_model']
        del state['env']

        # Restore everything else
        self.__dict__.update(state)


