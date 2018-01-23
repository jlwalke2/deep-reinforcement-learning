from collections import namedtuple
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation


class History(object):
    '''Performance monitor that handles logging and metric calculations.'''

    # TODO: Cleanup storage/handling of metrics as list of tuples/pandas
    # TODO: Allow tracking of step metrics to be turned on/off
    # TODO: Add optional checkpointing to a file
    def __init__(self, window_size=100):
        self.episode_metrics = []
        self.step_metrics = []
        self.episode_start_time = None
        self._EpisodeTuple = None
        self._StepTuple = None

    def on_step_end(self, **kwargs):
        if self._StepTuple is None:
            self._StepTuple = namedtuple('Step', kwargs.keys())

#        self.step_metrics.append(self._StepTuple(**kwargs))

    def on_episode_end(self, **kwargs):
        # Create a named tuple to match the fields if not already done
        if self._EpisodeTuple is None:
            self._EpisodeTuple = namedtuple('Episode', kwargs.keys())

        # Store the metrics
        self.episode_metrics.append(self._EpisodeTuple(**kwargs))

    def get_step_metrics(self):
        if self.step_metrics is None:
            return pd.DataFrame()

        metrics = pd.DataFrame(self.step_metrics)

        if 'step' in metrics.columns:
            metrics.set_index('step', inplace=True)

        return metrics

    def get_episode_metrics(self, start=0, end=None):
        assert end is None or end > start, 'Start record must occur before end record.'

        if self.episode_metrics is None:
            return pd.DataFrame()

        if isinstance(self.episode_metrics, pd.DataFrame):
            return self.episode_metrics

        end = end or len(self.episode_metrics)
        metrics = pd.DataFrame(self.episode_metrics[start:end])

        if 'episode' in metrics.columns:
            metrics.set_index('episode', inplace=False)
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
        if not filename.endswith('.h5'):
            filename += '.h5'
        with pd.HDFStore(filename, 'w') as store:
            store['episode_metrics'] = self.get_episode_metrics()

    @staticmethod
    def load(filename):
        if not filename.endswith('.h5'):
            filename += '.h5'

        with pd.HDFStore(filename, 'r') as store:
            obj = History()
            obj.episode_metrics = store.get('episode_metrics')
            t0 = obj.episode_metrics
            t1 = [a for a in dir(t0) if a.startswith('t')]
            if 'step_metrics' in store.keys():
                obj.step_metrics = store.get('step_metrics')

            return obj

