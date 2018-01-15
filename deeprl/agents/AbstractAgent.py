import gym
import gym.wrappers
import numpy as np
import logging
from shutil import rmtree
from tempfile import mkdtemp
from keras.models import Model, Sequential
from ..utils.monitor import Monitor
from abc import ABCMeta, abstractmethod
from EventHandler import EventHandler
from collections import namedtuple

Step = namedtuple('Step', ['s','a','r','s_prime','is_terminal'])

# TODO: Buffer Memory - for replay of whole trajectories (in order)
# TODO: Add methods for training start/stop, episode start/stop, etc
# TODO: Change exploration episodes to exploration steps?
# TODO: Implement frameskip / action replay
# TODO: Research keras callbacks for logging and metrics



class AbstractAgent:
    __metaclass__ = ABCMeta

    def __init__(self, env, policy=None, memory=None, max_steps_per_episode=0, logger=None, metrics=None, callbacks=[], name=None, api_key=None):
        self.name = name or self.__class__.__name__
        self.env = env
        self.policy = policy
        self.memory = memory
        self.max_steps_per_episode = max_steps_per_episode
        self.num_actions = AbstractAgent._get_space_size(env.action_space)
        self.num_features = AbstractAgent._get_space_size(env.observation_space)
        self.api_key = api_key

        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.INFO)

        # TODO:  Add easy way to turn on/off logging from different components
#        self.logger.parent.handlers[0].addFilter(logging.Filter('root.' + __name__))

        # Create a metrics object if one was not provided
        if metrics:
            self.metrics = metrics
            callbacks.append(self.metrics)
        else:
            metrics = [o for o in callbacks if isinstance(o, Monitor)]
            if len(metrics) == 0:
                self.metrics = Monitor()
                callbacks.append(self.metrics)
            else:
                self.metrics = metrics[0]

        self.episode_start = EventHandler()
        self.episode_end = EventHandler()
        self.step_start = EventHandler()
        self.step_end = EventHandler()
        self.train_start = EventHandler()
        self.train_end = EventHandler()
        self.calculate_error = EventHandler()       # Called as pre-process hook for error calculation

        # TODO: differentiate callbacks from preprocess steps

        # Automatically hook up any events
        for event in ['on_episode_start', 'on_episode_end', 'on_step_start', 'on_step_end', 'on_train_start',
                      'on_train_end', 'on_calculate_error']:
            handler = event.replace('on_', '')

            if handler in dir(self):
                handler = getattr(self, handler)
                for observer in [self, self.policy, self.memory, self.logger] + callbacks:
                    if event in dir(observer):
                        handler += getattr(observer, event)

        # if self.memory is not None:
        #     self.step_end += self._store_memory

        self._episode_end_template = 'Episode {episode_count}: \tError: {total_error:.2f} \tReward: {total_reward:.2f}'

    @abstractmethod
    def choose_action(self, state):
        raise NotImplementedError


    @staticmethod
    def _clone_model(model):
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
            total_steps = 0

            for episode_count in range(1, max_episodes + 1):
                self._raise_episode_start_event(episode_count=episode_count)
                # START of episode
                render = episode_count % render_every_n == 0
                episode_done = False
                total_episode_reward = 0
                total_episode_error = 0
                step_count = 0

                s = self.env.reset()  # Get initial state observation

                while not episode_done:
                    if render:
                        self.env.render()
                    s = np.asarray(s)

                    self._raise_step_start_event(episode_count=episode_count, step=step_count, total_steps=total_steps, s=s)

                    a = self.choose_action(s.reshape(1, -1))

                    s_prime, r, episode_done, _ = self.env.step(a)

                    # Some environments return reward as an array, flatten into a float for consistency
                    if isinstance(r, np.ndarray):
                        r = np.sum(r)

                    step_count += 1
                    total_steps += 1
                    total_episode_reward += r  # Track rewards without shaping

                    s, a, r, s_prime, episode_done = self.preprocess_state(s, a, r, s_prime, episode_done)

                    # Force the episode to end if we've reached the maximum number of steps allowed
                    if self.max_steps_per_episode and step_count >= self.max_steps_per_episode:
                        episode_done = True

                    if self.memory is not None:
                        self.memory.append((s, a, r, s_prime, episode_done))

                    self._raise_step_end_event(episode_count=episode_count, step=step_count, total_steps=total_steps, s=s, s_prime=s_prime, a=a, r=r,
                                               episode_done=episode_done)

                    s = s_prime

                self._raise_episode_end_event(episode_count=episode_count, total_reward=total_episode_reward, total_error=total_episode_error, episode_steps=step_count, total_steps=total_steps)

                total_steps += step_count

            self.env.close()

        except KeyboardInterrupt:
            return
        finally:
            if upload:
                gym.upload(monitor_path, api_key=self.api_key)
                rmtree(monitor_path) # Cleanup the temp dir


    # def _store_memory(self, **kwargs):
    #     for k in ['s','a','r','s_prime','episode_done']:
    #         assert k in kwargs, 'Key {} not found in parameters of step_end event.'.format(k)
    #
    #     if self.memory is not None:
    #         self.memory.append((kwargs['s'], kwargs['a'], kwargs['r'], kwargs['s_prime'], kwargs['episode_done']))

    def __raise_event(self, event, **kwargs):
        kwargs['sender'] = self.name
        event(**kwargs)

    def _raise_episode_start_event(self, **kwargs):
        self.__raise_event(self.episode_start, **kwargs)

    def _raise_episode_end_event(self, **kwargs):
        self.__raise_event(self.episode_end, **kwargs)

        if self._episode_end_template:
            self.logger.info(self._episode_end_template.format(**kwargs))

    def _raise_step_start_event(self, **kwargs):
        self.__raise_event(self.step_start, **kwargs)

    def _raise_step_end_event(self, **kwargs):
        self.__raise_event(self.step_end, **kwargs)

    def __raise_train_start_event(self, **kwargs):
        self.__raise_event(self.train_start, **kwargs)

    def __raise_train_end_event(self, **kwargs):
        self.__raise_event(self.train_end, **kwargs)
