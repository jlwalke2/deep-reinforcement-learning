import gym
import gym.wrappers
import numpy as np
import logging
import random as rnd


from shutil import rmtree
from tempfile import mkdtemp
from keras.models import Model, Sequential
from ..utils.monitor import Monitor

from EventHandler import EventHandler

# TODO: Add methods for training start/stop, episode start/stop, etc
# TODO: Change exploration episodes to exploration steps?
# TODO: Implement frameskip / action replay
# TODO: Research keras callbacks for logging and metrics


class AbstractAgent:
    def __init__(self, env, model, policy=None, memory=None, max_steps_per_episode=0, logger=None, callbacks=[], api_key=None, seed=None):
        self.env = env
        self.policy = policy
        self.model = model
        self.memory = memory
        self.max_steps_per_episode = max_steps_per_episode

        if isinstance(env.action_space, gym.spaces.discrete.Discrete):
            self.num_actions = env.action_space.n
        else:
            self.num_actions = env.action_space.shape[0]

        if isinstance(env.observation_space, gym.spaces.discrete.Discrete):
            self.num_features = env.observation_space.n
        else:
            self.num_features = env.observation_space.shape[0]

        self.api_key = api_key
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.INFO)

        # TODO:  Add easy way to turn on/off logging from different components
#        self.logger.parent.handlers[0].addFilter(logging.Filter('root.' + __name__))

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
                for observer in [self.policy, self.memory, self.logger] + callbacks:
                    if event in dir(observer):
                        handler += getattr(observer, event)

        if self.memory is not None:
            self.step_end += self._store_memory

        if seed is not None:
            env.seed(seed)
            rnd.seed(seed)
            np.random.seed(seed)


    def choose_action(self, state):
        raise NotImplementedError


    @staticmethod
    def _clone_model(model):
        if isinstance(model, Sequential):
            return Sequential.from_config(model.get_config())
        elif isinstance(model, Model):
            return Model.from_config(model.get_config())


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
                self.episode_start(episode_count=episode_count)

                render = episode_count % render_every_n == 0
                episode_done = False
                total_reward = 0
                total_error = 0
                step_count = 0

                s = self.env.reset()  # Get initial state observation

                while not episode_done:
                    if render:
                        self.env.render()
                    s = np.asarray(s)

                    self.step_start(step=step_count, total_steps=total_steps, s=s)

                    a = self.choose_action(s.reshape(1, -1))

                    s_prime, r, episode_done, _ = self.env.step(a)

                    # Some environments return reward as an array, flatten into a float for consistency
                    if isinstance(r, np.ndarray):
                        r = np.sum(r)

                    step_count += 1
                    total_steps += 1
                    total_reward += r  # Track rewards without shaping

                    s, a, r, s_prime, episode_done = self.preprocess_state(s, a, r, s_prime, episode_done)

                    # Force the episode to end if we've reached the maximum number of steps allowed
                    if self.max_steps_per_episode and step_count >= self.max_steps_per_episode:
                        episode_done = True

                    # self.memory.append((s, a, r, s_prime, episode_done))
                    #
                    # # Hard update of the target model every N steps
                    # if total_steps % target_model_update == 0:
                    #     self._update_target_model()
                    #
                    # # Soft update every step
                    # elif target_model_update < 1.0:
                    #     self._update_target_model(target_model_update)
                    #
                    # # Train model weights
                    # if total_steps > steps_before_training:
                    #     error = self._update_weights()
                    #     total_error += error

                    self.step_end(step=step_count, total_steps=total_steps, s=s, s_prime=s_prime, a=a, r=r,
                                  episode_done=episode_done)

                    s = s_prime

                self.episode_end(episode_count=episode_count, total_reward=total_reward, total_error=total_error, num_steps=step_count)

                total_steps += step_count

            self.env.close()

        except KeyboardInterrupt:
            return
        finally:
            if upload:
                gym.upload(monitor_path, api_key=self.api_key)
                rmtree(monitor_path) # Cleanup the temp dir


    def _store_memory(self, **kwargs):
        for k in ['s','a','r','s_prime','episode_done']:
            assert k in kwargs, 'Key {} not found in parameters of step_end event.'.format(k)

        if self.memory is not None:
            self.memory.append((kwargs['s'], kwargs['a'], kwargs['r'], kwargs['s_prime'], kwargs['episode_done']))