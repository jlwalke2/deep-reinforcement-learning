from collections import deque
from datetime import datetime
from functools import wraps

# Deep Reinforcement Learnign that Matters (https://arxiv.org/pdf/1709.06560.pdf)

def callback_return(*metrics: str):
    """Designates that a function should return a specific set of metrics.

    Function will always return a dictionary of metrics and corresponding values.

    """
    assert len(metrics) > 0

    def decorator(func):

        @wraps(func)
        def wrapper(*args, **kwargs):
            stats = func(*args, **kwargs)

            if len(metrics) == 1:
                return {metrics[0]: stats}

            # Convert the returned values into a dictionary of metrics
            if isinstance(stats, dict):
                result = {k: stats.get(k, 0) for k in metrics}
            else:
                result = dict.fromkeys(metrics, 0)

                if isinstance(stats, list) or isinstance(stats, tuple):
                    for i in range(min(len(metrics), len(stats))):
                        result[metrics[i]] = stats[i]

            return result

        wrapper.is_metric = True
        wrapper.return_values = metrics
        return wrapper

    return decorator


class EpisodeTime():
    """Calculate the total runtime of an episode."""

    def on_episode_start(self, **kwargs):
        self.start = datetime.now()

    @callback_return('episode_time')
    def on_episode_end(self, **kwargs):
        return datetime.now() - self.start

class CumulativeReward():
    """Compute the cumulative reward of an agent across all timesteps."""
    def __init__(self):
        self._value = 0

    @callback_return('cumulative_reward')
    def on_step_end(self, **kwargs):
        assert 'reward' in kwargs

        self._value += kwargs.get('reward', 0)

        return self._value

class EpisodeReturn():
    """Compute the total reward across all steps in a single episode."""
    def __init__(self):
        self._value = 0

    def on_episode_start(self, **kwargs):
        self._value = 0

    def on_step_end(self, **kwargs):
        assert 'reward' in kwargs
        self._value += kwargs.get('reward', 0)

    @callback_return('episode_return')
    def on_episode_end(self, **kwargs):
        return self._value


class RollingEpisodeReturn():
    """Compute the rolling average of recent episode rewards."""

    def __init__(self):
        self._queue = deque(maxlen=50)
        self._value = 0

    def on_episode_start(self, **kwargs):
        self._value = 0

    def on_step_end(self, **kwargs):
        assert 'reward' in kwargs

        self._value += kwargs.get('reward', 0)

    @callback_return('rolling_return')
    def on_episode_end(self, **kwargs):
        self._queue.append(self._value)
        return sum(self._queue) / float(len(self._queue))



