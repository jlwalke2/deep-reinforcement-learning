import gym
import gym.wrappers
from keras.models import Sequential
import logging
import numpy as np
import pandas as pd
import random as rnd
from shutil import rmtree
from tempfile import mkdtemp

class Agent:
    def __init__(self, env, model, policy, memory, api_key=None, seed=None):
        self.env = env
        self.model = model
        self.policy = policy
        self.memory = memory
        self.num_actions = env.action_space.n
        if isinstance(env.observation_space, gym.spaces.discrete.Discrete):
            self.num_features = env.observation_space.n
        else:
            self.num_features = env.observation_space.shape[0]

        self.api_key = api_key
        self.logger = logging.getLogger('root.' + __name__)
        self.logger.setLevel(logging.INFO)

        # TODO:  Add easy way to turn on/off logging from different components
#        self.logger.parent.handlers[0].addFilter(logging.Filter('root.' + __name__))

        if not seed is None:
            env.seed(seed)
            rnd.seed(seed)
            np.random.seed(seed)


    def addHandlers(self, handlers):
        for handler in handlers:
            self.logger.addHandler(handler)


class DeepQAgent(Agent):
    pass

class DoubleDeepQAgent(Agent):
    def __init__(self, env, model, policy, memory, gamma=0.9, api_key=None, seed=None):
        super().__init__(env, model, policy, memory, api_key, seed)

        self.target_model = Sequential.from_config(self.model.get_config())
        self.gamma = gamma
        self.preprocess_steps = []
        


    def train(self, max_episodes=500, steps_before_training=0, target_model_update=1000, render_every_n=1, upload=False, running_average_len=50):
        # TODO: Running avg length should be handled by logging/metrics framework

        if upload:
            assert self.api_key, 'An API key must be specified before uploading training results.'
            monitor_path = mkdtemp()
            self.env = gym.wrappers.Monitor(self.env, monitor_path)

        episode_completed_events = []
        if 'on_episode_complete' in dir(self.policy):
            episode_completed_events.append(self.policy.on_episode_complete)

        metrics = pd.DataFrame(dict(Error=np.zeros(max_episodes), Reward=np.zeros(max_episodes)))

        try:
            step_count = 0

            for episode_count in range(1, max_episodes + 1):
                render = episode_count % render_every_n == 0
                total_reward, total_error, steps = self._run_episode(target_model_update, steps_before_training, step_count, render)

                # Fire any notifications
                for event in episode_completed_events:
                    event()

                step_count += steps
                metrics.Reward[episode_count] = total_reward
                metrics.Error[episode_count] = total_error

                start = max(0, episode_count - running_average_len)
                avg_reward = sum(metrics.Reward[start:episode_count + 1]) / running_average_len
                self.logger.info('Episode {}: \tError: {},\tTotal Reward: {} \tMoving Avg Reward:{}\tBuffer: {}'.format(
                        episode_count, round(total_error, 2), total_reward, avg_reward, len(self.memory)))

            self.env.close()

        except KeyboardInterrupt:
            return
        finally:
            if upload:
                rmtree(monitor_path) # Cleanup the temp dir

        return metrics
    
    
    def _run_episode(self, target_model_update, steps_before_training, step_count, render):
        episode_done = False
        total_reward = 0
        total_error = 0

        s = self.env.reset()  # Get initial state observation
        
        # TODO: Terminate after max steps
        while not episode_done:
            if render:
                self.env.render()
            s = np.asarray(s)
            q_values = self.model.predict_on_batch(s.reshape(1,-1))
            a = self.policy(q_values)

            s_prime, r, episode_done, _ = self.env.step(a)

            step_count += 1
            total_reward += r  # Track rewards without shaping

            self.memory.append((s, a, r, s_prime, episode_done))

            # Hard update of the target model every N steps
            if step_count % target_model_update == 0:
                self._update_target_model()

            # Soft update every step
            elif target_model_update < 1.0:
                self._update_target_model(target_model_update)

            # Train model weights
            if step_count > steps_before_training:
                error = self._update_weights()
                total_error += error

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

        targets = self.model.predict_on_batch(states)
        targets[range(n), actions.astype('int32').ravel()] = updated_targets[:, 0]

        error = self.model.train_on_batch(states, targets)

        return error
