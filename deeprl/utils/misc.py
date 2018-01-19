import numpy as np
import os
import random as rnd
import keras.backend as K
import keras.models
from warnings import warn
import matplotlib.pyplot as plt
from matplotlib import animation

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