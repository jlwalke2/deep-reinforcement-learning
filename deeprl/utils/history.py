from collections import namedtuple
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
from ..memories import TrajectoryMemory
from ..policies import RandomPolicy
from PIL import Image

class History(object):
    '''Performance monitor that handles logging and metric calculations.'''

    # TODO: Cleanup storage/handling of metrics as list of tuples/pandas
    # TODO: Allow tracking of step metrics to be turned on/off
    # TODO: Add optional checkpointing to a file
    def __init__(self, window_size=100):
        self.episode_metrics = []
        self.step_metrics = []
        self.episode_start_time = None
        self._EpisodeTuple = None
        self._StepTuple = None

    def on_step_end(self, **kwargs):
        if self._StepTuple is None:
            self._StepTuple = namedtuple('Step', kwargs.keys())

#        self.step_metrics.append(self._StepTuple(**kwargs))

    def on_episode_end(self, **kwargs):
        # Create a named tuple to match the fields if not already done
        if self._EpisodeTuple is None:
            self._EpisodeTuple = namedtuple('Episode', kwargs.keys())

        # Store the metrics
        self.episode_metrics.append(self._EpisodeTuple(**kwargs))

    def get_step_metrics(self):
        if self.step_metrics is None:
            return pd.DataFrame()

        metrics = pd.DataFrame(self.step_metrics)

        if 'step' in metrics.columns:
            metrics.set_index('step', inplace=True)

        return metrics

    def get_episode_metrics(self, start=0, end=None):
        assert end is None or end > start, 'Start record must occur before end record.'

        if self.episode_metrics is None:
            return pd.DataFrame()

        if isinstance(self.episode_metrics, pd.DataFrame):
            return self.episode_metrics

        end = end or len(self.episode_metrics)
        metrics = pd.DataFrame(self.episode_metrics[start:end])

        if 'episode' in metrics.columns:
            metrics.set_index('episode', inplace=False)
        return metrics


    def get_episode_plot(self, columns, interval=1000):
        fig = plt.figure()
        axis = fig.add_subplot(1, 1, 1)

        def animate(i):
            data = self.get_episode_metrics()

            if data.shape[0] > 0:
                axis.clear()
                for col in columns:
                    axis.plot(data[col])

            return axis

        anim = animation.FuncAnimation(fig, animate, interval=interval)
        return plt, anim

    def save(self, filename):
        if not filename.endswith('.h5'):
            filename += '.h5'
        with pd.HDFStore(filename, 'w') as store:
            store['episode_metrics'] = self.get_episode_metrics()

    @staticmethod
    def load(filename):
        if not filename.endswith('.h5'):
            filename += '.h5'

        with pd.HDFStore(filename, 'r') as store:
            obj = History()
            obj.episode_metrics = store.get('episode_metrics')
            t0 = obj.episode_metrics
            t1 = [a for a in dir(t0) if a.startswith('t')]
            if 'step_metrics' in store.keys():
                obj.step_metrics = store.get('step_metrics')

            return obj


class RandomSample(object):
    """Generates a random sample of states from an environment."""

    def __init__(self, env):
        self.env = env
        self._data = None
        self.policy = RandomPolicy(env)
        self.thumbnails = []

    # Taken from: https://github.com/tensorflow/tensorflow/issues/6322
    def images_to_sprite(self, data):
        """Creates the sprite image along with any necessary padding

        Args:
          data: NxHxW[x3] tensor containing the images.

        Returns:
          data: Properly shaped HxWx3 image with any necessary padding.
        """
        if len(data.shape) == 3:
            data = np.tile(data[..., np.newaxis], (1, 1, 1, 3))
        data = data.astype(np.float32)
        min = np.min(data.reshape((data.shape[0], -1)), axis=1)
        data = (data.transpose(1, 2, 3, 0) - min).transpose(3, 0, 1, 2)
        max = np.max(data.reshape((data.shape[0], -1)), axis=1)
        data = (data.transpose(1, 2, 3, 0) / max).transpose(3, 0, 1, 2)
        # Inverting the colors seems to look better for MNIST
        # data = 1 - data

        n = int(np.ceil(np.sqrt(data.shape[0])))
        padding = ((0, n ** 2 - data.shape[0]), (0, 0),
                   (0, 0)) + ((0, 0),) * (data.ndim - 3)
        data = np.pad(data, padding, mode='constant',
                      constant_values=0)
        # Tile the individual thumbnails into an image.
        data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3)
                                                               + tuple(range(4, data.ndim + 1)))
        data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
        data = (data * 255).astype(np.uint8)
        return data

    def run(self, sample_size: int =1000, frame_skip: int =4, thumbnail_size=None):
        memory = TrajectoryMemory(maxlen=sample_size)
        s = self.env.reset()

        for step in range(sample_size * frame_skip):
            s = np.asarray(s)
            a = self.policy(s)
            s_prime, r, episode_done, _ = self.env.step(a)

            if step % frame_skip == 0:
                memory.append((s, a, r, s_prime, episode_done))

                if thumbnail_size is not None:
                    # TODO: Figure out how to render Env to array without displaying window
                    image = Image.fromarray(self.env.render(mode='rgb_array'))
                    self.thumbnails.append(np.asarray(image.resize(thumbnail_size)))

            if episode_done:
                s = self.env.reset()
            else:
                s = s_prime

        self._data = memory.sample()

    @property
    def actions(self):
        if self._data is None:
            raise ValueError('The `.run()` method must be called before sample data is available.')

        return self._data.actions

    @property
    def rewards(self):
        if self._data is None:
            raise ValueError('The `.run()` method must be called before sample data is available.')

        return self._data.rewards

    @property
    def states(self):
        if self._data is None:
            raise ValueError('The `.run()` method must be called before sample data is available.')

        return self._data.states

    @property
    def s_primes(self):
        if self._data is None:
            raise ValueError('The `.run()` method must be called before sample data is available.')

        return self._data.s_primes


    @property
    def sprite(self):
        sprite = self.images_to_sprite(np.asarray(self.thumbnails))
        return Image.fromarray(sprite)
