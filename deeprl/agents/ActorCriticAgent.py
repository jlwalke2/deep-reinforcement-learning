import keras.backend as K
from keras.models import Model, Sequential, Input
from keras.layers import Lambda
from .AbstractAgent import AbstractAgent
import numpy as np

# A3C https://arxiv.org/pdf/1602.01783.pdf


class ActorCriticAgent(AbstractAgent):
    def __init__(self, actor, critic, gamma=0.99, **kwargs):

        self.gamma = gamma

        super(ActorCriticAgent, self).__init__(**kwargs)

        '''
        Need two separate models (actor & critic)
        Each model has a separate target network (subclass DDQN?)


        actor input = state, output = action.  Assert actor output matches env action space
        critic input = state + action, output = q value.  Assert input matches actor input + output, Assert output is q-value


        '''

#        assert len(critic.input) == 2, 'Critic model must take two inputs: state and action.'
#        assert K.int_shape(actor.input) == K.int_shape(critic.input[0]), 'State input to actor in critic models does not match.'
#        assert K.int_shape(actor.output) == K.int_shape(critic.input[1]), 'Action input to critic model does not the output from the actor model.'

        # Create target models

        self.actor, self.trainable_actor = self._build_actor_model(actor)
        #self.actor_target = self._clone_model(self.actor)
        self.critic = critic
        self.critic_target = self._clone_model(self.critic)


    def _build_actor_model(self, model):
        # Return new model w/

        actions = model.output
        mask = Input(shape=(self.num_actions,), name='Mask')
        delta = Input(shape=(1,), name='DeltaV')

        def loss(inputs):
            actions, mask, delta = inputs
            return 1 * K.sum(K.log(actions) * mask, axis=-1) * delta

        loss_out = Lambda(loss, output_shape=(1,), name='LossCalc')([actions, mask, delta])
        inputs = [model.input, mask, delta]
        actor = Model(inputs=inputs, outputs=[loss_out])
        actor.compile(model.optimizer, loss=lambda y_true, y_pred: y_pred)
        return model, actor




    def train(self, target_model_update, **kwargs):
        """Train the agent in the environment for a specified number of episodes."""
        self._target_model_update = target_model_update
        self.step_end += self._update_model_weights
        self.step_end += self._update_target_weights

        # Run the training loop
        super().train(**kwargs)


    def choose_action(self, state):
        # Actor network returns probability of choosing each action in current state
        actions = self.actor.predict_on_batch(state)
        self.logger.debug(actions)
        # Select the action to take
        return np.random.choice(np.arange(actions.size), p=actions.ravel())


    def _update_model_weights(self, **kwargs):
        assert 'total_steps' in kwargs, 'Key "total_steps" not found in parameters of step_end event.'

        # Don't train until the minimum number of steps have been observed
        if kwargs['total_steps'] < self.steps_before_training:
            return

        states, actions, rewards, s_primes, flags = self.memory.sample()

        assert states.shape == (self.memory.sample_size, self.num_features)
        assert s_primes.shape == (self.memory.sample_size, self.num_features)
        assert actions.shape == (self.memory.sample_size, 1)
        assert rewards.shape == (self.memory.sample_size, 1)
        assert flags.shape == (self.memory.sample_size, 1)

        s_prime_val = self.critic_target.predict_on_batch(s_primes)                 # V(s')
        s_prime_val[flags.ravel()] = 0                                              # V(s') = 0 if s' is terminal
        critic_target = rewards + self.gamma * s_prime_val                          # V(s)
        critic_error = self.critic.train_on_batch(states, critic_target)            # update critic weights

        critic_pred = self.critic.predict_on_batch(states)
        delta = critic_target - critic_pred

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
            assert 'total_steps' in kwargs.keys(), 'Key "total_steps" not found'

            if kwargs['total_steps'] % self._target_model_update == 0:
                # Hard update
                self.critic_target.set_weights(self.critic.get_weights())
