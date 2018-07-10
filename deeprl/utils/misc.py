import logging
import numpy as np
import os
import random as rnd
import keras.backend as K
import keras.models
from warnings import warn
import matplotlib.pyplot as plt
from matplotlib import animation
from PIL import Image
from ..policies import RandomPolicy
from ..memories import TrajectoryMemory

logger = logging.getLogger(__name__)


class RandomSample(object):
    """Generates a random sample of states from an environment."""

    # TODO: Add support for max steps

    def __init__(self, env, thumbnail_size=None):
        self.env = env
        self._data = None
        self.policy = RandomPolicy(env)
        self.thumbnail_size = thumbnail_size
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
        if thumbnail_size:
            self.thumbnail_size = thumbnail_size

        memory = TrajectoryMemory(maxlen=sample_size)
        s = self.env.reset()

        logger.info(f'Randomly sampling {sample_size} states from the environment...')

        for step in range(sample_size * frame_skip):
            s = np.asarray(s)
            a = self.policy(s)
            s_prime, r, episode_done, _ = self.env.step(a)

            if step % frame_skip == 0:
                memory.append((s, a, r, s_prime, episode_done))


                if self.thumbnail_size is not None:
                    # TODO: Figure out how to render Env to array without displaying window
                    image = Image.fromarray(self.env.render(mode='rgb_array'))
                    self.thumbnails.append(np.asarray(image.resize(thumbnail_size)))

            if episode_done:
                s = self.env.reset()
            else:
                s = s_prime

        self._data = memory.sample()

        logger.info(f'Sampling complete.')

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


def unwrap_model(model):
    """Extract the underlying Model instance from a Keras class."""

    # Keras Sequential models have an underlying Model object with the correct input/output tensors.
    if hasattr(model, 'model'):
        model = model.model

    return model

def keras2dict(model):
    assert isinstance(model, keras.models.Model)

    config = dict(model_type=type(model),
                model_config=model.get_config())

    if hasattr(model, 'optimizer'):
        config['optimizer_type'] = type(model.optimizer)
        config['optimizer_config'] = model.optimizer.get_config()

    # Get loss if specified.  For Sequential models, need to unwrap and use inner Model
    config['loss'] = getattr(model, 'loss', None)
    if config['loss'] is None and hasattr(model, 'model'):
        config['loss'] = getattr(model.model, 'loss', None)

    config['weights'] = model.get_weights()

    return config



def dict2keras(config):
    model = config['model_type'].from_config(config['model_config'])

    if 'optimizer_type' in config:
        optimizer = config['optimizer_type'].from_config(config['optimizer_config'])
        model.compile(optimizer, config['loss'])

    model.set_weights(config['weights'])

    return model

def set_seed(seed, env=None):
    # See these references for further details:
    # https://docs.python.org/3.4/using/cmdline.html#envvar-PYTHONHASHSEED
    # https://github.com/keras-team/keras/issues/2280#issuecomment-306959926
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    rnd.seed(seed)
    if env:
        env.seed(seed)

    if K.backend() == 'tensorflow':
        import tensorflow as tf
        session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
        tf.set_random_seed(seed)
        sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
        K.set_session(sess)
        warn('A seed was specified.  Tensorflow will use a single thread to enable reproducible results.',
             category=RuntimeWarning)
    else:
        warn('Only able to set seeds when using the Tensorflow backend.  Results may not be reproducible.',
             category=RuntimeWarning)

# TODO: Group & plot by sender
def animated_plot(func, columns, interval=1000):
    '''
    Generate an animated plot using matplotlib.

    Display the plot using plt.show(block=False) to enable background updates during training.

    :param func: method that returns the data to plot
    :param columns: column names in the data to be used in the plot
    :param interval:
    :return:
    '''
    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1)

    def animate(i):
        data = func()

        if data.shape[0] > 0:
            axis.clear()
            for col in columns:
                axis.plot(data[col])

        return axis

    anim = animation.FuncAnimation(fig, animate, interval=interval)
    return plt, anim