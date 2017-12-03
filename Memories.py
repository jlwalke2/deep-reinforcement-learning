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

        elements = [np.asarray(x[i]) for i in range(len(x))]
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
    def __init__(self, maxlen=1000, sample_size=32):
        super().__init__(maxlen, sample_size)

    def append(self, x):
        x = list(x) + [np.finfo('float32').max]  # Initial error amount.  Max value?

        super().append(x)

    def sample(self, n=0):
        if not n:
            n = self.sample_size
        oversample = n > len(self)
        nrows = self.buffer.shape[0] if self.is_full else self.index

        probs = self.buffer[:nrows, -1].copy()
        probs -= - probs.min()
        probs /= probs.sum()

        indx = np.random.choice(nrows, n, replace=oversample, p=probs)

        states = self.buffer[indx, 0:self._field_splits[0]]
        actions = self.buffer[indx, self._field_splits[0]:self._field_splits[1]]
        rewards = self.buffer[indx, self._field_splits[1]:self._field_splits[2]]
        s_primes = self.buffer[indx, self._field_splits[2]:self._field_splits[3]]
        flags = self.buffer[indx, self._field_splits[3]:self._field_splits[4]].astype('bool_')

        return states, actions, rewards, s_primes, flags
