import keras.backend as K
from ..utils.callbacks import TensorBoardCallback
from keras.models import Model, Sequential, Input
from keras.layers import Lambda
from .abstract import AbstractAgent
import numpy as np

from ..utils.misc import unwrap_model

# A3C https://arxiv.org/pdf/1602.01783.pdf
import logging

logger = logging.getLogger(__name__)
np.seterr('raise')

class ActorCriticAgent(AbstractAgent):
    def __init__(self, actor, critic, **kwargs):

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

        self.actor_tensorboard = TensorBoardCallback('tensorboard/ac/actor', histogram_freq=1, write_grads=True)
        self.actor_tensorboard.set_model(unwrap_model(self.actor))
        self.actor_tensorboard.scalars += ['step', 'rolling_return']

        # self.critic_tensorboard = TensorBoardCallback('tensorboard/ac/critic', histogram_freq=1, write_grads=True)
        # self.critic_tensorboard.set_model(unwrap_model(self.critic))
        # self.critic_tensorboard.scalars += ['step', 'rolling_return', 's0_value']


    def _build_actor_model(self, model):
        # Return new model w/

        actions = model.output
        mask = Input(shape=(self.num_actions,), name='Mask')
        delta = Input(shape=(1,), name='DeltaV')

        def loss(inputs):
            actions, mask, delta = inputs

            return -1 * K.sum(K.log(actions + 1e-20) * mask, axis=-1) * delta


        loss_out = Lambda(loss, output_shape=(1,), name='LossCalc')([actions, mask, delta])
        inputs = [model.input, mask, delta]
        actor = Model(inputs=inputs, outputs=[loss_out])
        actor.compile(model.optimizer, loss=lambda y_true, y_pred: y_pred)
        return model, actor




    def train(self, target_model_update, **kwargs):
        """Train the agent in the environment for a specified number of episodes."""
        self._target_model_update = target_model_update
#        self.step_end += self._update_model_weights
#        self.step_end += self._update_target_weights
        self.steps_before_training = 32
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

#        s_prime_val = self.critic.predict_on_batch(s_primes)                 # V(s')
        s_prime_val = self.critic_target.predict_on_batch(s_primes)                 # V(s')
        s_prime_val[flags.ravel()] = 0                                              # V(s') = 0 if s' is terminal
        critic_target = rewards + self.gamma * s_prime_val                          # V(s)
        critic_error = self.critic.train_on_batch(states, critic_target)            # update critic weights

        critic_pred = self.critic.predict_on_batch(states)
        delta = critic_target - critic_pred + 1e-10

        mask = np.zeros((self.memory.sample_size, self.num_actions))
        mask[range(actions.shape[0]), actions.astype('int32').ravel()] = 1.0

        dummy_target = np.zeros_like(actions)

        actor_error = self.trainable_actor.train_on_batch([states, mask, delta], dummy_target)
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



    def on_execution_start(self, **kwargs):
        super().on_execution_start(**kwargs)
        self.actor_tensorboard.on_execution_start(**kwargs)
        # self.critic_tensorboard.on_execution_start(**kwargs)

    def on_warmup_start(self, **kwargs):
        self._warmup = True
        self._warmup_states = []
        self._warmup_actions = []
        self._warmup_values = []

    def on_warmup_end(self, **kwargs):
        self._warmup = False

        input = np.asarray(self._warmup_states).reshape((-1, self.num_features))
        actor_output = np.asarray(self._warmup_actions).reshape((-1, self.num_actions))
        critic_output = np.asarray(self._warmup_values).reshape((-1,1))
        actor_sample_weights = np.ones(input.shape[0])
        critic_sample_weights = np.ones(input.shape[0])

        self.actor_tensorboard.validation_data = [input, actor_output, actor_sample_weights]
        # self.critic_tensorboard.validation_data = [input, critic_output, critic_sample_weights]

    def on_step_end(self, **kwargs):
        super().on_step_end(**kwargs)

        # Store test data if we're in the warmup phase
        if self._warmup:
            self._warmup_states.append(kwargs['s'])
            self._warmup_actions.append(kwargs['actor_out'])
            self._warmup_values.append(kwargs['s_value'])

    def on_episode_end(self, **kwargs):
        super().on_episode_end(**kwargs)

        if not self._warmup:
            self.actor_tensorboard.on_episode_end(**kwargs)
            # self.critic_tensorboard.on_episode_end(**kwargs)

    def on_execution_end(self, **kwargs):
        super().on_execution_end(**kwargs)

        self.actor_tensorboard.on_execution_end()
        # self.critic_tensorboard.on_execution_end()
