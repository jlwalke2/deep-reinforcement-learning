from deeprl.utils.async import ModelManager
import multiprocessing
import os, os.path
import threading
from warnings import warn
from deeprl.utils import History
import logging, logging.handlers
import pickle
import matplotlib.pyplot as plt


# TODO: Don't generate agents until experiment is actually run
# TODO: Assume agents passed in along with num to create and func to call when process starts

def _run_agent(func, agent, history, queue):
    # Log all messages to the queue
    logger = logging.getLogger()

    # Delete existing loggers and force everything through the queue
    logger.handlers.clear()
    logger.addHandler(logging.handlers.QueueHandler(queue))
    logger.setLevel(logging.INFO)

    try:
        logger.info(f'Process {multiprocessing.current_process().name} is starting...')

        # TODO: EventHandlers won't be hooked up
        agent.history = history
        return func(agent)
    finally:
        logger.debug(f'Process {multiprocessing.current_process().name} terminated.')


def _run_queue_listener(queue):
    logger = logging.getLogger()

    # TODO: Take in configuration
    # TODO: Replace with QueueListener?
    logger.addHandler(logging.StreamHandler())

    while True:
        record = queue.get()

        if record == None:      # Signal from main thread to terminate logging loop
            break
        logger.handle(record)


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

        listener = threading.Thread(target=_run_queue_listener, args=(log_queue,))
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
                    multiprocessing.Process(target=_run_agent, args=(func, agent, history, log_queue), name=agent.name))

        for p in processes:
            p.start()

        for p in processes:
            p.join()

        log_queue.put_nowait(None)  # Signal to listener to terminate thread
        listener.join()
        history.save(self._HISTORY_PATH)


    def get_plot(self, df, metric: str, **kwargs: 'passed to Pandas .plot()'):
        df.pivot(columns='sender', values=metric).groupby(
            lambda colname: colname.rsplit('_', maxsplit=1)[0], axis=1).mean().plot(**kwargs)


    def get_plots(self, metrics: list, shape: tuple =None):
        if shape is None:
            shape = (1, len(metrics))
        else:
            assert shape[0] * shape[1] >= len(metrics), 'Subplot layout of {} does not subplots for {} metrics.'.format(shape, len(metrics))

        history = History.load(self._HISTORY_PATH)
        episode_df = history.get_episode_metrics().set_index('episode')

        fig, axes = plt.subplots(nrows=shape[0], ncols=shape[1], figsize=(8, 4))
        for i in range(len(metrics)):
            self.get_plot(episode_df, metrics[i], ax=axes[i])

        return fig
