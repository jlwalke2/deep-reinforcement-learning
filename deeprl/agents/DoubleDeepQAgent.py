import numpy as np
from keras.models import Sequential
from .DeepQAgent import DeepQAgent


class DoubleDeepQAgent(DeepQAgent):
    def __init__(self, **kwargs):
        super(DoubleDeepQAgent, self).__init__(**kwargs)

        self.target_model = Sequential.from_config(self.model.get_config())

    def choose_action(self, state):
        q_values = self.model.predict_on_batch(state)
        return self.policy(q_values)

    def train(self, target_model_update, **kwargs):
        if target_model_update <= 0:
            raise ValueError('target_model_update must be a positive number.')

        self._target_model_update = target_model_update

        super(DoubleDeepQAgent, self).train(**kwargs)


    def _run_episode(self, target_model_update, steps_before_training, total_steps, render):
        episode_done = False
        total_reward = 0
        total_error = 0
        step_count = 0
        self._target_model_update = target_model_update
        self._steps_before_training = steps_before_training

        s = self.env.reset()  # Get initial state observation

        while not episode_done:
            if render:
                self.env.render()
            s = np.asarray(s)

            self.step_start(step=step_count, total_steps=total_steps, s=s)

            q_values = self.model.predict_on_batch(s.reshape(1, -1))
            a = self.policy(q_values)

            s_prime, r, episode_done, _ = self.env.step(a)

            step_count += 1
            total_steps += 1
            total_reward += r  # Track rewards without shaping

            s, a, r, s_prime, episode_done = self.preprocess_state(s, a, r, s_prime, episode_done)

            # Force the episode to end if we've reached the maximum number of steps allowed
            if self.max_steps_per_episode and step_count >= self.max_steps_per_episode:
                episode_done = True

            self.memory.append((s, a, r, s_prime, episode_done))

            # Hard update of the target model every N steps
            if total_steps % target_model_update == 0:
                self._update_target_model()

            # Soft update every step
            elif target_model_update < 1.0:
                self._update_target_model(target_model_update)

            # Train model weights
            if total_steps > steps_before_training:
                error = self._update_weights()
                total_error += error

            self.step_end(step=step_count, total_steps=total_steps, s=s, s_prime=s_prime, a=a, r=r,
                          episode_done=episode_done)
            self.logger.debug("S: {}\tA: {}\tR: {}\tS': {}\tDone: {}".format(s, a, r, s_prime, episode_done))

            s = s_prime


        return total_reward, total_error, step_count

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
        targets[range(n), actions.astype('int32').ravel()] = updated_targets[:, 0]

        self.calculate_error(target=targets, estimate=estimate)

        diff = targets[range(n), actions.astype('int32').ravel()] - updated_targets[:, 0]
        # targets[range(n), actions.astype('int32').ravel()] = updated_targets[:, 0]

        error = self.model.train_on_batch(states, targets)

        self.train_end(num_samples=n, s=states, a=actions, r=rewards, s_prime=s_primes, episode_done=flags,
                       target=targets, delta=diff, error=error)

        return error



    # def on_episode_end(self, **kwargs):
    #     self.logger.info(self._episode_end_template.format(**kwargs))

    def on_step_end(self, **kwargs):
        assert 'total_steps' in kwargs
        total_steps = kwargs.get('total_steps')

        # Perform hard / soft updates of target model weights
        if self._target_model_update < 1.0:
            self._update_target_model(self._target_model_update)
        elif total_steps % self._target_model_update == 0:
            self._update_target_model()

        if total_steps > self._steps_before_training:
            error = self._update_weights()
#            total_error += error


