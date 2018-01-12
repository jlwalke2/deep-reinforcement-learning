import numpy as np
import os
import random as rnd
import keras.backend as K
from warnings import warn

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
        sess = tf.Session(graph=tf.get_default_graph(),
                          config=tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1))
        K.set_session(sess)
        warn('A seed was specified.  Tensorflow will use a single thread to enable reproducible results.',
             category=RuntimeWarning)
    else:
        warn('Only able to set seeds when using the Tensorflow backend.  Results may not be reproducible.',
             category=RuntimeWarning)