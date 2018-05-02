import gym, gym.spaces
import keras.backend as K
from keras.models import Input
import numpy as np
from functools import reduce
import operator
import warnings


try:
    import tensorflow as tf
except ImportError:
    pass

from .abstract import AbstractAgent
from ..utils.misc import unwrap_model, RandomSample


import logging

logger = logging.getLogger(__name__)
np.seterr('raise')

def get_array_size(shape):
    return reduce(operator.mul, (x for x in shape if x is not None))

def clip_gradient(gradients, clipvalue=0, clipnorm=0):
    if clipvalue > 0:
        if isinstance(gradients, list):
            gradients = [K.clip(grad, -clipvalue, clipvalue) for grad in gradients]
        else:
            gradients = K.clip(gradients, -clipvalue, clipvalue)
    elif clipnorm > 0:
        from keras.optimizers import clip_norm

        if isinstance(gradients, list):
            norm = K.sqrt(sum([K.sum(K.square(grad)) for grad in gradients]))
            gradients = [clip_norm(grad, clipnorm, norm) for grad in gradients]
        else:
            pass #TODO: Implement

    return gradients

class ActorCriticAgent(AbstractAgent):
    def __init__(self, actor, critic, **kwargs):
        """
        Implementation of the Deep Deterministic Policy Gradient (DDPG) algorithm. See:
            Continuous Control with Deep Reinforcement Learning
            https://arxiv.org/pdf/1509.02971.pdf

        Suitable for environments with continuous action spaces.  Observation space
        may be discrete continuous.

        """
        super(ActorCriticAgent, self).__init__(**kwargs)

        if K.backend() != 'tensorflow':
            raise RuntimeError(f'{type(self)} currently only supports the TensorFlow backend.')

        # DDPG implementation only supports continuous action spaces.
        if not isinstance(self.env.action_space, gym.spaces.Box):
            raise TypeError(f'DDPG agent expects an environment with a continuous action space.  Received {type(env.action_space)} instead.')

        # Convert tensor dimensions to total number of input values
        actor_input_size = get_array_size(K.int_shape(actor.inputs[0]))
        actor_output_size = get_array_size(K.int_shape(actor.outputs[0]))

        # Validate actor architecture
        assert len(actor.inputs) == 1, f'Expected an actor network that takes a single input: the current state.  Received an actor with {len(actor.inputs)} inputs.'
        assert len(actor.outputs) == 1, f'Expected an actor network with a single output tensor: the action to take.  Received an actor with {len(actor.outputs)} outputs.'
        assert actor_input_size == self.num_features, f"Actor network's input tensor with shape of {actor.inputs[0].shape} does not match state dimensions of {self.num_features}."

        # TODO: Fix.  Check dimension of action, not number of actions
        assert actor_output_size == self.num_actions, f"Actor network's output tensor with shape of {actor.outputs[0].shape} does not match state dimensions of {self.num_actions}."

        t0 = actor.layers[0]
        t1 = actor.layers[0](critic.inputs[0])

        if critic is None:
            pass
            # TODO: Create critic based on actor

        # Validate critic architecture
        assert len(critic.inputs) == 2, f'Expected a critic network that takes two inputs: the state and the action.  Received a critic with {len(critic.inputs)} inputs.'
        assert K.int_shape(critic.inputs[0]) == K.int_shape(actor.inputs[0]), f'Critic state input with shape {critic.inputs[0].shape} does not match expected dimensions of {actor.inputs[0].shape}.'
        assert K.int_shape(critic.inputs[1]) == K.int_shape(actor.outputs[0]), f'Critic action input with shape {critic.inputs[1].shape} does not match expected dimensions of {actor.outputs[0].shape}.'

        actor_args = {}
        if getattr(actor, 'optimizer', None) is not None:
            msg = f'Actor has been compiled with {type(actor.optimizer)}, but will not be used during training'
            warnings.warn(msg, stacklevel=2)
            logger.warning(msg)

            for attr in ['clipnorm', 'clipvalue', 'lr']:
                actor_args[attr] = getattr(actor.optimizer, attr, 0)

        self.train_actor = self._get_actor_train_function(actor, **actor_args)
        self.train_critic = critic.train_on_batch

        # Clone models to create target models
        self.actor = actor
        self.actor_target = self._clone_model(actor)
        self.critic = critic
        self.critic_target = self._clone_model(critic)

        self._grad_q_wrt_a = tf.gradients(critic.output, critic.inputs[1])

        # Clip gradient Q(s, a) wrt a if critic model has gradient clipping set
        if getattr(critic, 'optimizer', None) is not None:
            args = {}
            for attr in ['clipnorm', 'clipvalue']:
                args[attr] = getattr(critic.optimizer, attr, 0)

            self._grad_q_wrt_a = clip_gradient(self._grad_q_wrt_a, **args)

        self._metric_names.update({'error','delta', 'q_value', 'actor_out'})


    def _configure_tensorboard(self, path):
        from ..utils.callbacks import TensorBoardCallback

        self.validation_data = RandomSample(gym.make(self.env.spec.id))
        self.validation_data.run(sample_size=100, thumbnail_size=(100, 75))

        self.actor_tensorboard = TensorBoardCallback(path + '/actor', histogram_freq=1, write_grads=True)
        self.actor_tensorboard.set_model(unwrap_model(self.actor))
        self.actor_tensorboard.scalars += ['step', 'rolling_return']
        self.add_callbacks(self.actor_tensorboard)

        input = self.validation_data.states
        actions = self.actor.predict_on_batch(self.validation_data.states)
        sample_weights = np.ones(input.shape[0])
        self.actor_tensorboard.validation_data = [input, actions, sample_weights]


        # self.critic_tensorboard = TensorBoardCallback(path + '/critic', histogram_freq=1, write_grads=True)
        # self.critic_tensorboard.set_model(unwrap_model(self.critic))
        # self.add_callbacks(self.critic_tensorboard)




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

    def _get_actor_train_function(self, actor, lr=1e-4, clipnorm=0, clipvalue=0):
        # Placeholder for input of gradient Q(s, a) wrt a
        grad_q_wrt_a = Input(shape=(self.num_actions,), name='grad_q_input')

        # Compute gradient of actor weights wrt action * gradient of Q wrt action
        # By chain rule, gives gradient of Q wrt actor weights
        # NOTE: using TensorFlow since K.gradient(x, y) does not support passing gradient of ys
        actor_grads = tf.gradients(actor.outputs[0], actor.trainable_weights, -grad_q_wrt_a)

        actor_grads = clip_gradient(actor_grads, clipvalue=clipvalue, clipnorm=clipnorm)

        updates = zip(actor_grads, actor.trainable_weights)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr).apply_gradients(updates)
        K.get_session().run(tf.global_variables_initializer())

        def train_func(state, grad):
            K.get_session().run(optimizer, feed_dict={
                actor.inputs[0]: state,
                grad_q_wrt_a: grad
            })

        return train_func




    def train(self, target_model_update, **kwargs):
        """Train the agent in the environment for a specified number of episodes."""
        self._target_model_update = target_model_update

        self.steps_before_training = 32     # TODO: Should not be hardcoded. Reuse warmup steps

        # Run the training loop
        super().train(**kwargs)


    def choose_action(self, state):
        # For DDPG actor network returns the action to take
        action = self.actor.predict_on_batch(state)
        action = np.asarray(action).reshape((1, -1))

        # Policy is still used to add noise for exploration
        action = self.policy(action)

        self._status['actor_out'] = action
        self._status['q_value'] = self.critic.predict_on_batch([state, action])

        assert not np.any(np.isnan(action)), 'NaN value(s) output by the Actor network: {}'.format(action)

        return action



    def _update_weights(self):
        stats = self._update_model_weights()
        self._update_target_weights()
        return stats

    def _update_model_weights(self, **kwargs):
        assert 'total_steps' in self._status

        # Only proceed if the initial exploration steps have been completed.
        if self._status.total_steps <= self.steps_before_training:
            return

        states, actions, rewards, s_primes, flags = self.memory.sample()

        assert states.shape == (self.memory.sample_size, self.num_features)
        assert s_primes.shape == (self.memory.sample_size, self.num_features)
        assert actions.shape == (self.memory.sample_size, self.num_actions)
        assert rewards.shape == (self.memory.sample_size, 1)
        assert flags.shape == (self.memory.sample_size, 1)

        s_prime_actions = self.actor_target.predict_on_batch(s_primes)              # pi(s')
        predicted_s_prime_value = self.critic_target.predict_on_batch([s_primes, s_prime_actions])  # Predicted Q(s', a')
        observed_q_value = self.gamma * predicted_s_prime_value
        observed_q_value[flags] = 0.                # Return(s') = 0 if s' is a terminal state
        observed_q_value += rewards

        # Compute gradient of Q(s, a) wrt a.
        grad_tf = K.get_session().run(self._grad_q_wrt_a, feed_dict={
            self.critic.inputs[0]: states,
            self.critic.inputs[1]: actions
        })[0]  # Returns list of 1 gradient

        # Update TF actor weights
        self.train_actor(states, grad_tf)
        err_critic = self.train_critic([states, actions], observed_q_value)

        diff = observed_q_value - self.critic.predict_on_batch([states, actions])

        return dict(total_error=err_critic, delta=diff)



        # TODO: Report error metrics


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


