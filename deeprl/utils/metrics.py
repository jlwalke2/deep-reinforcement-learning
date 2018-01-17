from collections import deque

def metric(object):
    """Designates an object as a calculator for a metric."""
    object.is_metric = True
    if not hasattr(object, 'name'):
        object.name = object.__name__
    return object

@metric
class CumulativeReward():
    """Compute the cumulative reward of an agent across all timesteps."""
    def __init__(self):
        self._value = 0

    def on_step_end(self, **kwargs):
        assert 'reward' in kwargs

        self._value += kwargs.get('reward', 0)

        return self._value

@metric
class EpisodeReward():
    """Compute the total reward across all steps in a single episode."""
    def __init__(self):
        self._value = 0

    def on_episode_start(self, **kwargs):
        self._value = 0

    def on_step_end(self, **kwargs):
        assert 'reward' in kwargs
        self._value += kwargs.get('reward', 0)

    def on_episode_end(self, **kwargs):
        return self._value


@metric
class RollingEpisodeReward():
    """Compute the rolling average of recent episode rewards."""

    def __init__(self, window=50):
        self.name = 'RollingEpisodeReward{}'.format(window)
        self._queue = deque(maxlen=window)
        self._value = 0

    def on_episode_start(self, **kwargs):
        self._value = 0

    def on_step_end(self, **kwargs):
        assert 'reward' in kwargs

        self._value += kwargs.get('reward', 0)

    def on_episode_end(self, **kwargs):
        self._queue.append(self._value)
        return sum(self._queue) / float(len(self._queue))



