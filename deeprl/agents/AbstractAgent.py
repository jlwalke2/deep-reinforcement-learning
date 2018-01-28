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
from ..utils.metrics import EpisodeReturn, RollingEpisodeReturn, EpisodeTime

Step = namedtuple('Step', ['s','a','r','s_prime','is_terminal'])
logger = logging.getLogger(__name__)

# TODO: Reconcile logging teplates with metric names
# TODO: Change exploration episodes to exploration steps?
# TODO: Generate static initial experiences, states, etc for use in virtual batch normalization and q-value plots


"""Names of the events that the agent will raise.  Functions on callbacks with these names will be auto-wired up."""
_AGENT_EVENTS = {'on_episode_start', 'on_episode_end', 'on_step_start', 'on_step_end', 'on_train_start',
                      'on_train_end', 'on_execution_start', 'on_execution_end', 'on_warmup_start', 'on_warmup_end'}


class Status(dict):
    """Standard dictionary but set of keys cannot be changed after instantiation.
    Supports dict.key in addition to dict['key'].
    """
    def __init__(self, *vars):
        super(Status, self).__init__({v: None for v in vars})

    def __getattr__(self, item):
        return self[item]

    def __setattr__(self, key, value):
        if key.startswith('_'):
            super().__setattr__(key, value)
        elif key in self:
            self[key] = value
        else:
            raise KeyError('Unable to set value. Key {} does not exist in dictionary.'.format(key))

    def __setitem__(self, key, value):
        if key in self:
            super().__setitem__(key, value)
        else:
            raise KeyError('Unable to set value. Key {} does not exist in dictionary.'.format(key))

    def __getstate__(self):
        return self.__dict__.copy()

    def __setstate__(self, state):
        self.__dict__.update(state)


class AbstractAgent:
    __metaclass__ = ABCMeta

    def _build_status(self):
        metric_names = set()

        # Pull names from all registered metric callbacks
        for event in _AGENT_EVENTS:
            handler = event.replace('on_', '')

            if handler in dir(self):
                handler = getattr(self, handler)
                for event in handler.metrics:
                    metric_names.add(*event.return_values)

        # Add in names that the agent sets automatically
        metric_names.update(self._metric_names)

        return Status(*metric_names)



    def __init__(self, env, policy=None, memory=None, max_steps_per_episode=0, history=None, metrics=[], callbacks=[], name: str =None, api_key: str =None):
        self.name = name or self.__class__.__name__

        """Names of metrics that the agent computes directly and publishes.  This is in addition to any metrics returned by callbacks."""
        self._metric_names = {'episode','render','episode_done','reward','action','step','total_steps'}
        self.env = env
        self.policy = policy
        self.memory = memory
        self.max_steps_per_episode = max_steps_per_episode
        self.num_actions = AbstractAgent._get_space_size(env.action_space)
        self.num_features = AbstractAgent._get_space_size(env.observation_space)
        self.api_key = api_key

        # Setup default metrics if none were provided
        if len(metrics) == 0:
            metrics += [EpisodeReturn(), RollingEpisodeReturn(), EpisodeTime()]

        # Metrics are just events that return a value when called
        callbacks.extend(metrics)

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
        self.execution_start = EventHandler()
        self.warmup_start = EventHandler()
        self.warmup_end = EventHandler()
        self.episode_start = EventHandler()
        self.step_start = EventHandler()
        self.step_end = EventHandler()
        self.train_start = EventHandler()
        self.train_end = EventHandler()
        self.episode_end = EventHandler()
        self.execution_end = EventHandler()

        # Automatically hook up any events
        for observer in [self, self.policy, self.memory, logger] + callbacks:
            self.wire_events(observer)

        # Logging templates
        self.step_end_template = None
        self.episode_end_template = 'Episode {episode}: \tError: {total_error:.2f} \tReward: {episode_return: .2f} RollingEpisodeReward: {rolling_return: .2f} Runtime: {episode_time}'

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

        if hasattr(self, '_status'):
            self._status['sender'] = value

    @property
    def history(self):
        return self._history

    @history.setter
    def history(self, obj):
        if hasattr(self, '_history'):
            self.unwire_events(self._history)
        self._history = obj
        self.wire_events(obj)


    def unwire_events(self, observer):
        for event in _AGENT_EVENTS:
            handler = event.replace('on_', '')

            if handler in dir(self):
                handler = getattr(self, handler)
                if event in dir(observer):
                    handler -= getattr(observer, event)


    def wire_events(self, observer):
        for event in _AGENT_EVENTS:
            handler = event.replace('on_', '')

            if handler in dir(self):
                handler = getattr(self, handler)
                if event in dir(observer):
                    t0 = getattr(observer, event)
                    handler += getattr(observer, event)


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

    def test(self, num_episodes, frame_skip: int =1, render_every_n: int = 1, upload=False):
        # TODO: save & restore memory, history, callbacks, status

        self._exec_loop(num_episodes=num_episodes,
                        frame_skip=frame_skip,
                        warmup_steps=0,
                        render_every_n=render_every_n,
                        upload=upload,
                        train=False)

    def train(self, max_episodes: int =500, steps_before_training: int =None, frame_skip: int =4, render_every_n: int =0, upload: bool =False):
        """Train the agent in the environment for a specified number of episodes.

        :param max_episodes: Terminate training after this many new episodes are observed
        :param steps_before_training: Individual steps to take in the environment before training begins
        :param frame_skip:  Agent will only observe every n states.  The previously selected action will be used for all unobserved states.
        :param render_every_n: Render every nth episode
        :param upload: Upload training results to OpenAI Gym site?
        """

        # Unless otherwise specified, assume training doesn't start until a full sample of steps is observed
        if steps_before_training is None:
            steps_before_training = self.memory.sample_size

        self._exec_loop(num_episodes=max_episodes,
                        frame_skip=frame_skip,
                        warmup_steps=steps_before_training,
                        render_every_n=render_every_n,
                        upload=upload,
                        train=True)

    def _exec_loop(self, num_episodes: int, frame_skip: int, warmup_steps: int, render_every_n: int, train: bool, upload: bool):
        """Train the agent in the environment for a specified number of episodes.

        :param num_episodes: Terminate training after this many new episodes are observed
        :param steps_before_training: Individual steps to take in the environment before training begins
        :param render_every_n: Render every nth episode
        :param upload: Upload training results to OpenAI Gym site?
        """

        # TODO: pass metrics during training
        # TODO: wire & unwire events

        self._status = self._build_status()
        self._raise_execution_start_event()

        if upload:
            assert self.api_key, 'An API key must be specified before uploading training results.'
            monitor_path = mkdtemp()
            self.env = gym.wrappers.Monitor(self.env, monitor_path)

        self._raise_warmup_start_event()

        # If 0 or None is passed, disable rendering
        if not render_every_n:
            render_every_n = num_episodes + 1
        try:
            self._status.total_steps = 0

            for episode_count in range(1, num_episodes + 1):

                self._status.episode = episode_count
                self._status.render  = episode_count % render_every_n == 0
                self._status.episode_done = False
                self._status.step = 0
                total_episode_error = 0

                self._raise_episode_start_event()

                s = self.env.reset()  # Get initial state observation

                while not self._status.episode_done:
                    # End the warmup period as soon as the required number of steps have been taken
                    if self._status.total_steps == warmup_steps:
                        self._raise_warmup_end_event()

                    s = np.asarray(s)

                    self._raise_step_start_event(s=s)

                    # Action replay.  We repeat the selected action for n steps.
                    a = self.choose_action(s.reshape(1, -1))
                    r = 0

                    # Replay the selected action as necessary
                    # Accumulate the reward from each action, but don't let the agent observe the intermediate states
                    for _ in range(frame_skip):
                        if self._status.render:
                            self.env.render()

                        # Take action and observe reward and new state
                        s_prime, r_new, episode_done, _ = self.env.step(a)

                        # Some environments return reward as an array, flatten into a float for consistency
                        if isinstance(r_new, np.ndarray):
                            r_new = np.sum(r_new)

                        r += r_new

                        if episode_done:
                            break

                    self._status.action = a
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

                    # Train the agent's model(s) if necessary
                    if train:
                        self._raise_train_start_event()

                        stats = self._update_weights()
                        assert isinstance(stats, dict) or stats is None, \
                            'Value of {} returned by _update_weights() is not a dictionary or None'.format(stats)

                        if stats is not None:
                            self._status.update(stats)

                        self._raise_train_end_event()

                    s = s_prime


                self._raise_episode_end_event(total_error=total_episode_error)


            self.env.close()

        except KeyboardInterrupt:
            return
        finally:
            if upload:
                gym.upload(monitor_path, api_key=self.api_key)
                rmtree(monitor_path) # Cleanup the temp dir

            self._raise_execution_end_event()





    def _update_weights(self):
        """
        Called after the end of each step to allow the agent to perform any training updates.

        :return:  A dictionary of values to be added to _status or None
        """
        raise NotImplementedError

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

    def _raise_train_start_event(self, **kwargs):
        self.__raise_event(self.train_start, **kwargs)

    def _raise_train_end_event(self, **kwargs):
        self.__raise_event(self.train_end, **kwargs)

    def _raise_execution_start_event(self, **kwargs):
        self.__raise_event(self.execution_start, **kwargs)

    def _raise_execution_end_event(self, **kwargs):
        self.__raise_event(self.execution_end, **kwargs)

    def _raise_warmup_start_event(self, **kwargs):
        self.__raise_event(self.warmup_start, **kwargs)

    def _raise_warmup_end_event(self, **kwargs):
        self.__raise_event(self.warmup_end, **kwargs)

    def on_warmup_start(self, **kwargs):
        pass

    def on_warmup_end(self, **kwargs):
        pass

    def on_step_start(self, **kwargs):
        pass

    def on_step_end(self, **kwargs):
        logger.debug(self._status)

        if self.step_end_template:
            logger.info(self.step_end_template.format(**kwargs))

    def on_episode_start(self, **kwargs):
        pass

    def on_episode_end(self, **kwargs):
        if self.episode_end_template:
            logger.info(self.episode_end_template.format(**kwargs))

    def on_execution_start(self, **kwargs):
        pass

    def on_execution_end(self, **kwargs):
        pass