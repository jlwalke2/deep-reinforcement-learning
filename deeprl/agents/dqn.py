import gym
import numpy as np
from ..utils.misc import keras2dict, dict2keras, unwrap_model
from .abstract import AbstractAgent
from ..utils.callbacks import TensorBoardCallback
from ..utils.misc import RandomSample
import logging

logger = logging.getLogger(__name__)


class DeepQAgent(AbstractAgent):
    def __init__(self, model, *args, **kwargs):
        super(DeepQAgent, self).__init__(*args, **kwargs)
        self.model = model
        self.preprocess_steps = []

    def train(self, steps_before_training: int = None, **kwargs):
        # Unless otherwise specified, assume training doesn't start until a full sample of steps is observed
        self._steps_before_training = steps_before_training or self.memory.sample_size

        super(DeepQAgent, self).train(**kwargs)


class DoubleDeepQAgent(DeepQAgent):

    def __init__(self, **kwargs):
        super(DoubleDeepQAgent, self).__init__(**kwargs)

        # Additional metric values that this agent will set
        self._metric_names.update({'error', 'delta', 'qvalues'})

        # Create the target model.  Same architecture as DeepQ agent, but separate weights
        self.target_model = self._clone_model(self.model)

    def choose_action(self, state):
        q_values = self.model.predict_on_batch(state)
        self._status['qvalues'] = q_values

        assert np.any(np.isnan(q_values)) == False, 'Q-Values may not be NaN: {}'.format(q_values)

        return self.policy(q_values)


    def train(self, target_model_update: float, **kwargs):
        if target_model_update <= 0:
            raise ValueError('target_model_update must be a positive number.')

        self._target_model_update = target_model_update

        super(DoubleDeepQAgent, self).train(**kwargs)


    def _configure_tensorboard(self, path):
        import tensorflow as tf

        self._tensorboard_callback = TensorBoardCallback(path, histogram_freq=1, write_grads=True)
        self._tensorboard_callback.set_model(unwrap_model(self.model))
        self._tensorboard_callback.scalars += ['step', 'rolling_return']
        self.add_callbacks(self._tensorboard_callback)

        self.validation_data = RandomSample(gym.make(self.env.spec.id))
        self.validation_data.run(sample_size=1000, thumbnail_size=(100, 75))

        q_values = self.model.predict_on_batch(self.validation_data.states)
        tf_q_values = tf.Variable(q_values, name='QValues')
        tf_states = tf.Variable(self.validation_data.states, name='States')

        values = np.max(q_values, axis=1)  # V(s) = max_a Q(s, a)
        actions = np.argmax(q_values, axis=1)  # Best (greedy) action

        self.tensorboard_metadata = {tf_states.name: dict(Value_0=values, Action_0=actions)}
        sprites = {tf_states.name: (self.validation_data.sprite, self.validation_data.thumbnail_size),
                   tf_q_values.name: (self.validation_data.sprite, self.validation_data.thumbnail_size)}

        self._tensorboard_callback.add_embeddings([tf_states, tf_q_values], self.tensorboard_metadata, sprites)

        input = self.validation_data.states
        output = q_values
        sample_weights = np.ones(input.shape[0])
        self._tensorboard_callback.validation_data = [input, output, sample_weights]


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


    def _update_weights(self):
        '''Randomly select experiences from the replay buffer to train on.'''

        assert self._status.episode is not None
        assert 'total_steps' in self._status

        # Only proceed if the initial exploration steps have been completed.
        if self._status.total_steps <= self._steps_before_training:
            return

        # Hard update of target model weights every N steps
        if self._status.total_steps % self._target_model_update == 0:
            self._update_target_model()
        elif self._target_model_update < 1:
            # Soft weight update after every step
            self._update_target_model(self._target_model_update)

        n = self.memory.sample_size
        states, actions, rewards, s_primes, flags = self.memory.sample(n)

        assert states.shape == (n, self.num_features)
        assert s_primes.shape == (n, self.num_features)
        assert actions.shape == (n, 1)
        assert rewards.shape == (n, 1)
        assert flags.shape == (n, 1)

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
        targets[range(n), actions.astype('int32').ravel()] = updated_targets[:,
                                                             0] + 1e-10  # Add small epsilon to avoid 0 error

        diff = targets[range(n), actions.astype('int32').ravel()] - updated_targets[:, 0]

        error = self.model.train_on_batch(states, targets)

        return dict(total_error=error, delta=diff)


    def on_execution_end(self, **kwargs):
        if self._train and hasattr(self, 'tensorboard_metadata'):
            # Get new q-value estimates for states
            q_values = self.model.predict_on_batch(self.validation_data.states)
            values = np.max(q_values, axis=1)       # V(s) = max_a Q(s, a)
            actions = np.argmax(q_values, axis=1)   # Best (greedy) action

            for embedding in self._tensorboard_callback.embeddings:
                name = embedding.tensor_name
                if name in self.tensorboard_metadata:
                    metadata = self.tensorboard_metadata[name]
                    metadata.update(dict(Value_1=values, Action_1=actions))
                    self._tensorboard_callback.write_metadata(metadata, embedding.metadata_path)


    def __getstate__(self):
        state = self.__dict__.copy()

        # Convert Keras models into dictionaries of serializable objects
        state['model'] = keras2dict(self.model)
        state['target_model'] = keras2dict(self.target_model)
        state['env'] = self.env.spec.id

        return state


    def __setstate__(self, state):
        # Rebuild the Keras models
        self.model = dict2keras(state['model'])
        self.target_model = dict2keras(state['target_model'])
        self.env = gym.make(state['env'])

        del state['model']
        del state['target_model']
        del state['env']

        # Restore everything else
        self.__dict__.update(state)
