import gym
import keras.backend as K
from keras.models import Model, Input
from keras.layers import Lambda
import numpy as np

from .abstract import AbstractAgent
from ..utils.misc import unwrap_model, RandomSample


import logging

logger = logging.getLogger(__name__)
np.seterr('raise')

class ActorCriticAgent(AbstractAgent):
    def __init__(self, actor, critic, **kwargs):
        """ http://www-anw.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf
        """
        super(ActorCriticAgent, self).__init__(**kwargs)

#        assert len(critic.input) == 2, 'Critic model must take two inputs: state and action.'
#        assert K.int_shape(actor.input) == K.int_shape(critic.input[0]), 'State input to actor in critic models does not match.'
#        assert K.int_shape(actor.output) == K.int_shape(critic.input[1]), 'Action input to critic model does not the output from the actor model.'

        self._metric_names.update({'error','delta', 's_value', 'actor_out'})

        # Create target models
        self.actor, self.trainable_actor = self._build_actor_model(actor)
        #self.actor_target = self._clone_model(self.actor)
        self.critic = critic
        self.critic_target = self._clone_model(self.critic)

    def _configure_tensorboard(self, path):
        from ..utils.callbacks import TensorBoardCallback

        self.validation_data = RandomSample(gym.make(self.env.spec.id))
        self.validation_data.run(sample_size=1000, thumbnail_size=(100, 75))

        self.actor_tensorboard = TensorBoardCallback(path + '/actor', histogram_freq=1, write_grads=True)
        self.actor_tensorboard.set_model(unwrap_model(self.actor))
        self.actor_tensorboard.validation_data
        self.actor_tensorboard.scalars += ['step', 'rolling_return']
        self.add_callbacks(self.actor_tensorboard)

        input = self.validation_data.states
        actions = self.actor.predict_on_batch(self.validation_data.states)
        sample_weights = np.ones(input.shape[0])
        self.actor_tensorboard.validation_data = [input, actions, sample_weights]



        # q_values = self.model.predict_on_batch(self.validation_data.states)
        # tf_q_values = tf.Variable(q_values, name='QValues')
        # tf_states = tf.Variable(self.validation_data.states, name='States')
        #
        # values = np.max(q_values, axis=1)  # V(s) = max_a Q(s, a)
        # actions = np.argmax(q_values, axis=1)  # Best (greedy) action
        #
        # self.tensorboard_metadata = {tf_states.name: dict(Value_0=values, Action_0=actions)}
        # sprites = {tf_states.name: (self.validation_data.sprite, self.validation_data.thumbnail_size),
        #            tf_q_values.name: (self.validation_data.sprite, self.validation_data.thumbnail_size)}
        #
        # self._tensorboard_callback.add_embeddings([tf_states, tf_q_values], self.tensorboard_metadata, sprites)



        # self.critic_tensorboard = TensorBoardCallback('tensorboard/ac/critic', histogram_freq=1, write_grads=True)
        # self.critic_tensorboard.set_model(unwrap_model(self.critic))
        # self.critic_tensorboard.scalars += ['step', 'rolling_return', 's0_value']


        # import tensorflow as tf
        #
        # self._tensorboard_callback = TensorBoardCallback(path, histogram_freq=1, write_grads=True)
        # self._tensorboard_callback.set_model(unwrap_model(self.model))
        # self._tensorboard_callback.scalars += ['step', 'rolling_return']
        # self.add_callbacks(self._tensorboard_callback)
        #
        # self.validation_data = RandomSample(gym.make(self.env.spec.id))
        # self.validation_data.run(sample_size=1000, thumbnail_size=(100, 75))
        #
        # q_values = self.model.predict_on_batch(self.validation_data.states)
        # tf_q_values = tf.Variable(q_values, name='QValues')
        # tf_states = tf.Variable(self.validation_data.states, name='States')
        #
        # values = np.max(q_values, axis=1)  # V(s) = max_a Q(s, a)
        # actions = np.argmax(q_values, axis=1)  # Best (greedy) action
        #
        # self.tensorboard_metadata = {tf_states.name: dict(Value_0=values, Action_0=actions)}
        # sprites = {tf_states.name: (self.validation_data.sprite, self.validation_data.thumbnail_size),
        #            tf_q_values.name: (self.validation_data.sprite, self.validation_data.thumbnail_size)}
        #
        # self._tensorboard_callback.add_embeddings([tf_states, tf_q_values], self.tensorboard_metadata, sprites)
        #
        # input = self.validation_data.states
        # output = q_values
        # sample_weights = np.ones(input.shape[0])
        # self._tensorboard_callback.validation_data = [input, output, sample_weights]


    def _build_actor_model(self, model):
        # Return new model w/

        actions = model.output
        mask = Input(shape=(self.num_actions,), name='Mask')
        delta = Input(shape=(1,), name='DeltaV')

        def loss(inputs):
            actions, mask, delta = inputs

            return -1 * K.sum(K.log(actions + 1e-20) * delta, axis=-1)
#            return -1 * K.sum(K.log(actions + 1e-20) * mask, axis=-1) * delta


        loss_out = Lambda(loss, output_shape=(1,), name='LossCalc')([actions, mask, delta])
        inputs = [model.input, mask, delta]
        actor = Model(inputs=inputs, outputs=[loss_out])
        actor.compile(model.optimizer, loss=lambda y_true, y_pred: y_pred)
        return model, actor




    def train(self, target_model_update, **kwargs):
        """Train the agent in the environment for a specified number of episodes."""
        self._target_model_update = target_model_update

        self.steps_before_training = 32     # TODO: Should not be hardcoded. Reuse warmup steps

        # Run the training loop
        super().train(**kwargs)


    def choose_action(self, state):
        # Actor network returns probability of choosing each action in current state
        actions = self.actor.predict_on_batch(state)
        self._status['actor_out'] = actions
        self._status['s_value'] = self.critic.predict_on_batch(state)

        assert not np.any(np.isnan(actions)), 'NaN value(s) output by the Actor network: {}'.format(actions)

        logger.debug(actions)
        actions = np.round(actions, 10)
        # Select the action to take
#        return  actions[0]
        return np.random.choice(np.arange(actions.size), p=actions.ravel())

    def _update_weights(self):
        self._update_model_weights()
        self._update_target_weights()

    def _update_model_weights(self, **kwargs):
        assert 'total_steps' in self._status

        # Only proceed if the initial exploration steps have been completed.
        if self._status.total_steps <= self.steps_before_training:
            return

        states, actions, rewards, s_primes, flags = self.memory.sample()

        assert states.shape == (self.memory.sample_size, self.num_features)
        assert s_primes.shape == (self.memory.sample_size, self.num_features)
        assert actions.shape == (self.memory.sample_size, 1)
        assert rewards.shape == (self.memory.sample_size, 1)
        assert flags.shape == (self.memory.sample_size, 1)

        s_prime_val = self.critic_target.predict_on_batch(s_primes)                 # V(s')
        s_prime_val[flags.ravel()] = 0                                              # V(s') = 0 if s' is terminal
        critic_target = rewards + self.gamma * s_prime_val                          # V(s) = r + gamma * V(s')

        critic_pred = self.critic_target.predict_on_batch(states)
        delta = critic_target - critic_pred + 1e-20

        critic_error = self.critic.train_on_batch(states, critic_target)            # update critic weights


        mask = np.zeros((self.memory.sample_size, self.num_actions))
        mask[range(actions.shape[0]), actions.astype('int32').ravel()] = 1.0

        dummy_target = np.zeros_like(actions)

        actor_error = self.trainable_actor.train_on_batch([states, mask, critic_target], dummy_target)
        #actor_error = self.trainable_actor.train_on_batch([states, mask, delta], dummy_target)
        # pred = true - log(a)*(R-v)

        pass

    def _update_target_weights(self, **kwargs):
        # Check target model update
        # Check if steps before training reached?
        # Update weights    def _update_target_model(self, ratio=None):

        if self._target_model_update < 1.:
            # Soft model update
            w_t = self.critic_target.get_weights()
            w_o = self.critic.get_weights()

            for layer in range(len(w_o)):
                w_t[layer] = self._target_model_update * w_o[layer] + (1.0 - self._target_model_update) * w_t[layer]
            self.critic_target.set_weights(w_t)
        else:
            # Check if steps before training reached
            assert 'total_steps' in self._status

            if self._status['total_steps'] % self._target_model_update == 0:
                # Hard update
                self.critic_target.set_weights(self.critic.get_weights())


