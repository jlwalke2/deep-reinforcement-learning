import gym
import gym.wrappers
import numpy as np
import logging
from shutil import rmtree
from tempfile import mkdtemp
from keras.models import Model, Sequential
from abc import ABCMeta, abstractmethod
from collections import namedtuple
from ..utils import History, EventHandler
from ..utils.metrics import EpisodeReward, RollingEpisodeReward, EpisodeTime

Step = namedtuple('Step', ['s','a','r','s_prime','is_terminal'])

# TODO: Reconcile logging teplates with metric names
# TODO: Change exploration episodes to exploration steps?
# TODO: Implement frameskip / action replay

class DotDict(dict):
    """Standard dictionary with support for accessing items with dot notation: dict.key instead of dict['key']"""
    def __getattr__(self, item):
        return self.get(item, None)

    def __setattr__(self, key, value):
        self[key] = value


class AbstractAgent:
    __metaclass__ = ABCMeta

    def __init__(self, env, policy=None, memory=None, max_steps_per_episode=0, logger=None, history=None, metrics=[], callbacks=[], name: str =None, api_key: str =None):
        self.name = name or self.__class__.__name__
        self.env = env
        self.policy = policy
        self.memory = memory
        self.max_steps_per_episode = max_steps_per_episode
        self.num_actions = AbstractAgent._get_space_size(env.action_space)
        self.num_features = AbstractAgent._get_space_size(env.observation_space)
        self.api_key = api_key
        self._status = DotDict(sender=self.name)

        # Setup default metrics if none were provided
        if len(metrics) == 0:
            metrics += [EpisodeReward(), RollingEpisodeReward(), EpisodeTime()]

        # Metrics are just events that return a value when called
        callbacks.extend(metrics)

        # Create a logger if one was not provided
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.INFO)

        # Create a metrics object if one was not provided
        if history:
            self.history = history
            callbacks.append(self.history)
        else:
            history = [o for o in callbacks if isinstance(o, History)]
            if len(history) == 0:
                self.history = History()
                callbacks.append(self.history)
            else:
                self.history = history[0]

        # Setup event handlers for callbacks and metrics
        self.episode_start = EventHandler()
        self.episode_end = EventHandler()
        self.step_start = EventHandler()
        self.step_end = EventHandler()
        self.train_start = EventHandler()
        self.train_end = EventHandler()
        self.calculate_error = EventHandler()       # Called as pre-process hook for error calculation


        # Automatically hook up any events
        for event in ['on_episode_start', 'on_episode_end', 'on_step_start', 'on_step_end', 'on_train_start',
                      'on_train_end', 'on_calculate_error']:
            handler = event.replace('on_', '')

            if handler in dir(self):
                handler = getattr(self, handler)
                for observer in [self, self.policy, self.memory, self.logger] + callbacks:
                    if event in dir(observer):
                        handler += getattr(observer, event)

        # Logging templates
        self.step_end_template = None
        self.episode_end_template = 'Episode {episode}: \tError: {total_error:.2f} \tReward: {EpisodeReward: .2f} RollingEpisodeReward: {RollingEpisodeReward50: .2f} Runtime: {EpisodeTime.microseconds}'

    @abstractmethod
    def choose_action(self, state):
        raise NotImplementedError


    def _clone_model(self, model: Model):
        '''
        Clone an existing Keras model

        :return: A copy of the input model
        '''
        if isinstance(model, Sequential):
            return Sequential.from_config(model.get_config())
        elif isinstance(model, Model):
            return Model.from_config(model.get_config())

    @staticmethod
    def _get_space_size(env_space):
        '''
        Return the  size of an OpenAI Gym action space or observation space

        :param env_space: An instance of gym.spaces
        :return:
        '''
        if isinstance(env_space, gym.spaces.discrete.Discrete):
            return env_space.n
        else:
            return env_space.shape[0]


    def preprocess_state(self, s, a, r, s_prime, episode_done):
        '''
        Called after a new step in the environment, but before the observation is added to memory or used for training.
        Override to perform reward shaping
        '''
        return (s, a, r, s_prime, episode_done)


    def train(self, max_episodes=500, steps_before_training=None, render_every_n=1, upload=False):
        """Train the agent in the environment for a specified number of episodes.

        :param max_episodes: Terminate training after this many new episodes are observed
        :param steps_before_training: Individual steps to take in the environment before training begins
        :param render_every_n: Render every nth episode
        :param upload: Upload training results to OpenAI Gym site?
        """
        if upload:
            assert self.api_key, 'An API key must be specified before uploading training results.'
            monitor_path = mkdtemp()
            self.env = gym.wrappers.Monitor(self.env, monitor_path)

        # Unless otherwise specified, assume training doesn't start until a full sample of steps is observed
        if steps_before_training is None:
            self.steps_before_training = self.memory.sample_size

        try:
            self._status.total_steps = 0

            for episode_count in range(1, max_episodes + 1):
                self._status.episode = episode_count
                self._status.render  = episode_count % render_every_n == 0
                self._status.episode_done = False
                self._status.step = 0
                total_episode_error = 0

                self._raise_episode_start_event()

                s = self.env.reset()  # Get initial state observation

                while not self._status.episode_done:
                    if self._status.render:
                        self.env.render()
                    s = np.asarray(s)

                    self._raise_step_start_event(s=s)

                    a = self.choose_action(s.reshape(1, -1))

                    s_prime, r, episode_done, _ = self.env.step(a)

                    # Some environments return reward as an array, flatten into a float for consistency
                    if isinstance(r, np.ndarray):
                        r = np.sum(r)

                    self._status.step += 1
                    self._status.reward = r
                    self._status.total_steps += 1
                    self._status.reward = r

                    s, a, r, s_prime, episode_done = self.preprocess_state(s, a, r, s_prime, episode_done)
                    self._status.episode_done = episode_done

                    # Force the episode to end if we've reached the maximum number of steps allowed
                    if self.max_steps_per_episode and self._status.step >= self.max_steps_per_episode:
                        self._status.episode_done = True

                    if self.memory is not None:
                        self.memory.append((s, a, r, s_prime, self._status.episode_done))

                    self._raise_step_end_event(s=s, s_prime=s_prime, a=a, r=r)

                    s = s_prime

                self._raise_episode_end_event(total_error=total_episode_error)


            self.env.close()

        except KeyboardInterrupt:
            return
        finally:
            if upload:
                gym.upload(monitor_path, api_key=self.api_key)
                rmtree(monitor_path) # Cleanup the temp dir


    def __raise_event(self, event, **kwargs):
        # Ensure status information is passed to eventhandlers
        kwargs.update(self._status)

        # Call events and store any metrics returned
        metrics = event(**kwargs)

        # Update the status with the computed metrics
        self._status.update(metrics)

    def _raise_episode_start_event(self, **kwargs):
        self.__raise_event(self.episode_start, **kwargs)

    def _raise_episode_end_event(self, **kwargs):
        self.__raise_event(self.episode_end, **kwargs)

    def _raise_step_start_event(self, **kwargs):
        self.__raise_event(self.step_start, **kwargs)

    def _raise_step_end_event(self, **kwargs):
        self.__raise_event(self.step_end, **kwargs)

    def __raise_train_start_event(self, **kwargs):
        self.__raise_event(self.train_start, **kwargs)

    def __raise_train_end_event(self, **kwargs):
        self.__raise_event(self.train_end, **kwargs)


    def on_step_end(self, **kwargs):
        self.logger.debug(self._status)

        if self.step_end_template:
            self.logger.info(self.step_end_template.format(**kwargs))

    def on_episode_end(self, **kwargs):
        if self.episode_end_template:
            self.logger.info(self.episode_end_template.format(**kwargs))
