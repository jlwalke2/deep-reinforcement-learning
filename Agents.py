import gym
import gym.wrappers
from keras.models import Sequential
import logging
import numpy as np
import random as rnd
from shutil import rmtree
from tempfile import mkdtemp
from EventHandler import EventHandler
from collections import deque
from datetime import datetime
import pandas as pd


# TODO: Change exploration episodes to exploration steps?

class Monitor(logging.getLoggerClass()):
    '''Performance monitor that handles logging and metric calculations.'''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.episode_metrics = []
        self.recent_rewards = deque(maxlen=50)

    def on_episode_start(self, *args, **kwargs):
        # Save start time so we can calculate episode duration later
        self.episode_start_time = datetime.now()

    def on_episode_end(self, *args, **kwargs):
        self.episode_metrics.append(kwargs)

        self.recent_rewards.append(kwargs['total_reward'])
        avg_reward = sum(self.recent_rewards) / len(self.recent_rewards)

        episode_duration = datetime.now() - self.episode_start_time

        self.info('Episode {}: \tError: {},\tReward: {} \tMoving Avg Reward:{}\tSteps: {}\tDuration: {}'.format(
            round(kwargs['episode_count'], 2),
            round(kwargs['total_error'], 2),
            round(kwargs['total_reward'], 2),
            round(avg_reward, 2),
            kwargs['num_steps'],
            episode_duration))

    def on_step_start(self, *args, **kwargs):
        pass

    def on_step_end(self, *args, **kwargs):
        pass

    def on_train_start(self, *args, **kwargs):
        pass

    def on_train_end(self, *args, **kwargs):
        pass

    def get_episode_metrics(self):
        metrics = {}

        # Convert from list of dictionaries to dictionary of lists
        for k in self.episode_metrics[0].keys():
            metrics[k] = [d[k] for d in self.episode_metrics]

        df = pd.DataFrame(metrics)
        df['mean_reward'] = pd.rolling_mean(df['total_reward'], window=50, min_periods=0)
        return df



class Agent:
    def __init__(self, env, model, policy, memory, max_steps_per_episode=0, logger=None, api_key=None, seed=None):
        self.env = env
        self.model = model
        self.policy = policy
        self.memory = memory
        self.max_steps_per_episode = max_steps_per_episode

        self.num_actions = env.action_space.n
        if isinstance(env.observation_space, gym.spaces.discrete.Discrete):
            self.num_features = env.observation_space.n
        else:
            self.num_features = env.observation_space.shape[0]

        self.api_key = api_key
        if logger:
            self.logger = logger
        else:
            logging.setLoggerClass(Monitor)
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.INFO)

        # TODO:  Add easy way to turn on/off logging from different components
#        self.logger.parent.handlers[0].addFilter(logging.Filter('root.' + __name__))

        self.episode_start = EventHandler()
        self.episode_end = EventHandler()
        self.step_start = EventHandler()
        self.step_end = EventHandler()
        self.train_start = EventHandler()
        self.train_end = EventHandler()

        # Automatically hook up any events
        for event in ['on_episode_start', 'on_episode_end', 'on_step_start', 'on_step_end', 'on_train_start', 'on_train_end']:
            handler = event.replace('on_', '')

            if handler in dir(self):
                handler = getattr(self, handler)
                for observer in [self.policy, self.memory, self.logger]:
                    if event in dir(observer):
                        handler += getattr(observer, event)

        if not seed is None:
            env.seed(seed)
            rnd.seed(seed)
            np.random.seed(seed)


    def preprocess_state(self, s, a, r, s_prime, episode_done):
        '''
        Called after a new step in the environment, but before the observation is added to memory or used for training.
        Override to perform reward shaping
        '''
        return (s, a, r, s_prime, episode_done)


class DeepQAgent(Agent):
    def __init__(self, gamma=0.9, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.gamma = gamma
        self.preprocess_steps = []


    def train(self, max_episodes=500, steps_before_training=None, target_model_update=1000, render_every_n=1, upload=False, running_average_len=50):
        if upload:
            assert self.api_key, 'An API key must be specified before uploading training results.'
            monitor_path = mkdtemp()
            self.env = gym.wrappers.Monitor(self.env, monitor_path)

        # Unless otherwise specified, assume training doesn't start until a full sample of steps is observed
        if steps_before_training is None:
            steps_before_training = self.memory.sample_size

        try:
            total_steps = 0

            for episode_count in range(1, max_episodes + 1):
                self.episode_start(episode_count=episode_count)

                render = episode_count % render_every_n == 0
                total_reward, total_error, steps_in_episode = self._run_episode(target_model_update, steps_before_training, total_steps, render)

                # Fire any notifications
                self.episode_end(episode_count=episode_count, total_reward=total_reward, total_error=total_error, num_steps=steps_in_episode)

                total_steps += steps_in_episode

            self.env.close()

        except KeyboardInterrupt:
            return
        finally:
            if upload:
                gym.upload(monitor_path, api_key=self.api_key)
                rmtree(monitor_path) # Cleanup the temp dir


class DoubleDeepQAgent(DeepQAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.target_model = Sequential.from_config(self.model.get_config())

    
    def _run_episode(self, target_model_update, steps_before_training, total_steps, render):
        episode_done = False
        total_reward = 0
        total_error = 0
        step_count = 0

        s = self.env.reset()  # Get initial state observation
        
        # TODO: Terminate after max steps
        while not episode_done:
            if render:
                self.env.render()
            s = np.asarray(s)

            self.step_start(step=step_count, total_steps=total_steps, s=s)

            q_values = self.model.predict_on_batch(s.reshape(1,-1))
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

            self.step_end(step=step_count, total_steps=total_steps, s=s, s_prime=s_prime, a=a, r=r, episode_done=episode_done)
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

        targets = self.model.predict_on_batch(states)
        # TODO: Add callback w/ diff for priority replay
        diff = targets[range(n), actions.astype('int32').ravel()] - updated_targets[:, 0]
        targets[range(n), actions.astype('int32').ravel()] = updated_targets[:, 0]

        error = self.model.train_on_batch(states, targets)

        self.train_end(num_samples=n, s=states, a=actions, r=rewards, s_prime=s_primes, episode_done=flags, target=targets, delta=diff, error=error)

        return error
