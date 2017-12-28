import numpy as np


class Memory():
    '''Represents an experience replay buffer as a Numpy array'''
    def __init__(self, maxlen=1000, sample_size=32):
        self.buffer = None                  # Lazy initialization since we don't know # of columns
        self.max_len = maxlen
        self.is_full = False
        self.index = 0
        self.sample_size = sample_size

    def append(self, x):

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

        return states, actions, rewards, s_primes, flags


    def __len__(self):
        if self.is_full:
            return self.max_len
        else:
            return self.index


class PrioritizedMemory(Memory):
    '''
    Memory buffer implementing Prioritized Experience Replay

    See https://arxiv.org/pdf/1511.05952.pdf
    '''
    def __init__(self, maxlen=1000, sample_size=32, alpha=0.6):
        super().__init__(maxlen, sample_size)

        self.alpha = alpha
        self.beta = 1.0
        self.epsilon = 1.0
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

        return states, actions, rewards, s_primes, flags

    def on_train_end(self, *args, **kwargs):
        if 'delta' in kwargs:
            assert self.last_sample is not None

            self.buffer[self.last_sample, -1] = np.abs(kwargs['delta'])
            pass

    def on_calculate_error(self, *args, **kwargs):
        delta = kwargs['target'] - kwargs['estimate']
        p = delta.sum(axis=-1) # Sum along rows since q-values only differ for the action taken
        p = np.power(np.abs(p) + self.epsilon, self.alpha)
        self.buffer[self.last_sample, -1] = p

        w = np.power(1.0 / len(self) * 1.0 / p, self.beta)
        w /= w.max()

        # Element wise multiplication works since delta is 0 for any action not taken
        # Reshape to allow broadcasting across delta matrix
        kwargs['target'][:, :] = kwargs['estimate'] + delta * w.reshape((-1, 1))
        pass
