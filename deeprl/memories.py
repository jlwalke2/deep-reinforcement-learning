import numpy as np
from collections import namedtuple

Memories = namedtuple('Step', ['states', 'actions', 'rewards', 's_primes', 'terminal'])

class Memory():
    '''Experience replay buffer for reinforcement learning.

    Stores experiences in a Numpy array and allows uniform random sampling of the experiences.
    '''
    def __init__(self, maxlen=1000, sample_size=32):
        '''

        :param maxlen: Maximum number of experiences to store
        :param sample_size: Number of experiences to sample unless otherwise specified
        '''
        self.buffer = None                  # Lazy initialization since we don't know # of columns
        self.max_len = int(maxlen)
        self.is_full = False
        self.index = 0
        self.sample_size = sample_size

    def append(self, x):
        '''
        Add a new experience to the memory
        :param x: List of: s, a, r, s' done
        :return: None
        '''
        elements = [np.asarray(x[i]).flatten() for i in range(len(x))]
        row = np.hstack(elements)
        row = row.reshape(1, -1)

        if self.buffer is None:
            self._field_splits = [elements[i].size for i in range(len(elements))]
            self._field_splits = np.cumsum(self._field_splits)
            self.buffer = np.zeros((self.max_len, row.shape[1]))

        self.buffer[self.index] = row
        self.index += 1

        if self.index == self.max_len:
            self.index = 0
            self.is_full = True


    def sample(self, n=0):
        '''
        Returns a random sample of previously observed experiences.
        Sampling is done with replacement only if the number of samples to be drawn exceeds
        the number of experiences curently in the memory.

        :param n: Number of experiences to sample
        :return: Numpy arrays: s, a, r, s', done
        '''
        if not n:
            n = self.sample_size
        oversample = n > len(self)

        if self.is_full:
            indx = np.random.choice(self.buffer.shape[0], n, replace=oversample)
        else:
            indx = np.random.choice(self.index, n, replace=oversample)

        states = self.buffer[indx, 0:self._field_splits[0]]
        actions = self.buffer[indx, self._field_splits[0]:self._field_splits[1]]
        rewards = self.buffer[indx, self._field_splits[1]:self._field_splits[2]]
        s_primes = self.buffer[indx, self._field_splits[2]:self._field_splits[3]]
        flags = self.buffer[indx, self._field_splits[3]:].astype('bool_')

        return Memories(states, actions, rewards, s_primes, flags)


    def __len__(self):
        if self.is_full:
            return self.max_len
        else:
            return self.index


class TrajectoryMemory(Memory):
    '''
    Experience reply buffer designed to hold the samples for a single trajectory.
    All experiences are returned, in order, instead of randomly sampled.
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.sample_size = self.max_len

    def sample(self):
        i = len(self)
        states = self.buffer[:i, 0:self._field_splits[0]]
        actions = self.buffer[:i, self._field_splits[0]:self._field_splits[1]]
        rewards = self.buffer[:i, self._field_splits[1]:self._field_splits[2]]
        s_primes = self.buffer[:i, self._field_splits[2]:self._field_splits[3]]
        flags = self.buffer[:i, self._field_splits[3]:].astype('bool_')

        # Overwrite existing rows with new rows instead of re-initializing buffer
        self.index = 0

        return Memories(states, actions, rewards, s_primes, flags)





class PrioritizedMemory(Memory):
    '''
    Memory buffer implementing Prioritized Experience Replay

    See Prioritized Experience Replay paper by Schaul et. al. [https://arxiv.org/pdf/1511.05952.pdf]
    '''
    def __init__(self, alpha=0.6, **kwargs):
        '''

        :param alpha: Value between 0 and 1 indicating how greedy the prioritization is, with 0 being uniform random sampling
        :param kwargs:
        '''
        super().__init__(**kwargs)
        assert alpha >= 0 and alpha <= 1, 'Alpha value of {} is not between 0 and 1.'.format(alpha)

        self.alpha = alpha
        self.beta = 1.0
        self.epsilon = 1e-4             # Small value added to error to avoid division by zero
        self.last_sample = None

    def append(self, x):
        x = list(x) + [np.finfo('float32').max]  # Initial error amount

        super().append(x)

    def sample(self, n=0):
        if not n:
            n = self.sample_size
        oversample = n > len(self)
        nrows = self.buffer.shape[0] if self.is_full else self.index

        delta = np.abs(self.buffer[:nrows, -1]) + self.epsilon
        probs = np.power(delta, self.alpha)
        probs /= probs.sum()

        indx = np.random.choice(nrows, n, replace=oversample, p=probs)
        self.last_sample = indx # Save so we can update errors later

        states = self.buffer[indx, 0:self._field_splits[0]]
        actions = self.buffer[indx, self._field_splits[0]:self._field_splits[1]]
        rewards = self.buffer[indx, self._field_splits[1]:self._field_splits[2]]
        s_primes = self.buffer[indx, self._field_splits[2]:self._field_splits[3]]
        flags = self.buffer[indx, self._field_splits[3]:self._field_splits[4]].astype('bool_')

        return Memories(states, actions, rewards, s_primes, flags)

    def on_train_end(self, *args, **kwargs):
        if 'delta' in kwargs and kwargs['delta'] is not None:
            assert self.last_sample is not None

            self.buffer[self.last_sample, -1] = np.abs(kwargs['delta'].ravel())

    def on_calculate_error(self, *args, **kwargs):

        # TODO: Fix.  Not currently hooked up (on_calculate_error removed)
        delta = kwargs['target'] - kwargs['estimate']
        p = delta.sum(axis=-1) # Sum along rows since q-values only differ for the action taken
        p = np.power(np.abs(p) + self.epsilon, self.alpha)
        self.buffer[self.last_sample, -1] = p

        w = np.power(1.0 / len(self) * 1.0 / p, self.beta)
        w /= w.max()

        # Element wise multiplication works since delta is 0 for any action not taken
        # Reshape to allow broadcasting across delta matrix
        kwargs['target'][:, :] = kwargs['estimate'] + delta * w.reshape((-1, 1))

