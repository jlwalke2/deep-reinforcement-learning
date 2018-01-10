from collections import deque, namedtuple
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation


def animated_plot(func, columns, interval=1000):
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

class Monitor(object):
    '''Performance monitor that handles logging and metric calculations.'''

    def __init__(self, window_size=100):
        self.episode_metrics = None
        self.recent_rewards = deque(maxlen=window_size)
        self.episode_start_time = None

    def on_episode_start(self, *args, **kwargs):
        # Save start time so we can calculate episode duration later
        self.episode_start_time = datetime.now()

    def compute_metrics(self, **kwargs):
        if kwargs.get('total_reward', None):
            self.recent_rewards.append(kwargs['total_reward'])
            kwargs['avg_reward'] = avg_reward = sum(self.recent_rewards) / len(self.recent_rewards)

        if self.episode_start_time:
            kwargs['episode_duration'] = datetime.now() - self.episode_start_time

        return kwargs

    def on_episode_end(self, **kwargs):
        # Suppliment with additional computed metrics
        metrics = self.compute_metrics(**kwargs)

        # Create a named tuple to match the fields if not already done
        if self.episode_metrics is None:
            self.episode_metrics = []
            global episode
            episode = namedtuple('Episode', metrics.keys())

        # Store the metrics
        self.episode_metrics.append(episode(**metrics))


    def get_episode_metrics(self):
        return pd.DataFrame(self.episode_metrics)


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