from collections import namedtuple
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation


class History(object):
    '''Performance monitor that handles logging and metric calculations.'''

    # TODO: Add optional checkpointing to a file
    def __init__(self, window_size=100):
        self.episode_metrics = []
        self.episode_start_time = None
        self._Tuple = None

    def on_episode_start(self, *args, **kwargs):
        # Save start time so we can calculate episode duration later
        self.episode_start_time = datetime.now()


    def on_episode_end(self, **kwargs):
        # Create a named tuple to match the fields if not already done
        if self._Tuple is None:
            self._Tuple = namedtuple('Episode', kwargs.keys())

        # Store the metrics
        self.episode_metrics.append(self._Tuple(**kwargs))

    def get_episode_metrics(self, start=0, end=None):
        assert end is None or end > start, 'Start record must occur before end record.'

        if self.episode_metrics is None:
            return pd.DataFrame()

        end = end or len(self.episode_metrics)
        metrics = pd.DataFrame(self.episode_metrics[start:end])
        metrics.set_index('episode_count', inplace=True)
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
        df = self.get_episode_metrics()
        df.to_csv(filename)