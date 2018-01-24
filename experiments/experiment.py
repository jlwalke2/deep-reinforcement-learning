from deeprl.utils.async import ModelManager
import multiprocessing
import os, os.path
from warnings import warn
from deeprl.utils import History
import logging, logging.handlers
import pickle
import matplotlib.pyplot as plt


def _run_agent(func, agent, queue):
    # Get the root logger in the agent's process
    logger = logging.getLogger()

    # Delete existing loggers and force everything through the queue
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.handlers.QueueHandler(queue))

    # Run the agent
    try:
        logger.info(f'Process {multiprocessing.current_process().name} is starting...')
        return func(agent)
    finally:
        logger.info(f'Process {multiprocessing.current_process().name} terminated.')



class Experiment(object):
    def __init__(self,
                 name: str,
                 agents: list):
        """Define an experiment to be executed.

        :param name: user-friendly name for the experiment.  Results will also be stored in a subdir with this name.
        """

        self.name = name

        # Assume agent is list of (agent, num_copies, init_func)
        self._agents = agents

        self._HISTORY_PATH = f'{self.name}/history.h5'


    def run(self, force_rerun: bool =False):

        if force_rerun and os.path.exists(self.name):
            os.rmdir(self.name)

        # If the results directory and the history data is present, then don't rerun
        if os.path.isdir(self.name) and os.path.exists(self._HISTORY_PATH):
            warn(f'Experiment not run because {os.path.abspath(self._HISTORY_PATH)} already exists.')
            return

        # Create a directory to store results in
        if not os.path.isdir(self.name):
            os.mkdir(self.name)

        manager = ModelManager()
        manager.start()
        history = manager.Monitor()
        log_queue = manager.Queue()

        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter('[%(asctime)s|%(levelname)s] %(message)s'))
        listener = logging.handlers.QueueListener(log_queue, handler, respect_handler_level=True)
        listener.start()

        processes = []
        for agent, num_copies, func in self._agents:
            # Clone the agent N times
            config = pickle.dumps(agent)
            for i in range(num_copies):
                instance = pickle.loads(config)
                instance.name += f'_{i}'
                instance.history = history

                # Fork separate processes for each agent to run in
                processes.append(
                    multiprocessing.Process(target=_run_agent, args=(func, instance, log_queue), name=instance.name))

        for p in processes:
            p.start()

        for p in processes:
            p.join()

        listener.stop()
        history.save(self._HISTORY_PATH)


    def get_plot(self, df, metric: str, intervals: bool =True, **kwargs: 'passed to Pandas .plot()'):
        if 'title' not in kwargs:
            kwargs['title'] = metric

        df = df.pivot(columns='sender', values=metric).groupby(lambda colname: colname.rsplit('_', maxsplit=1)[0], axis=1)

        mean = df.mean()
        p = mean.plot(**kwargs)

        if intervals:
            min = df.min()
            max = df.max()
            for col in mean.columns:
                p.fill_between(mean.index, min[col], max[col], alpha=0.2)


    def get_plots(self, metrics: list, shape: tuple =None):
        if shape is None:
            shape = (1, len(metrics))
        else:
            assert shape[0] * shape[1] >= len(metrics), 'Subplot layout of {} does not subplots for {} metrics.'.format(shape, len(metrics))

        history = History.load(self._HISTORY_PATH)
        episode_df = history.get_episode_metrics().set_index('episode', inplace=False)

        fig, axes = plt.subplots(nrows=shape[0], ncols=shape[1], figsize=(8, 4))
        for i in range(len(metrics)):
            self.get_plot(episode_df, metrics[i], ax=axes[i], title=metrics[i])

        return fig
