import logging
from collections import deque
from datetime import datetime
import pandas as pd


class Monitor(object):
    '''Performance monitor that handles logging and metric calculations.'''

    def __init__(self):
        self.episode_metrics = []
        self.recent_rewards = deque(maxlen=100)
        self.episode_start_time = None

    def on_episode_start(self, *args, **kwargs):
        # Save start time so we can calculate episode duration later
        self.episode_start_time = datetime.now()

    def on_episode_end(self, *args, **kwargs):
        self.episode_metrics.append(kwargs)

        self.recent_rewards.append(kwargs['total_reward'])
        avg_reward = sum(self.recent_rewards) / len(self.recent_rewards)

        if self.episode_start_time:
            episode_duration = datetime.now() - self.episode_start_time
        else:
            self.warning('Episode Start Time is unknown.  Has on_episode_start been called?')
            episode_duration = 0

        self.info('Episode {}: \tError: {},\tReward: {} \tMoving Avg Reward:{}\tSteps: {}\tDuration: {}'.format(
            round(kwargs['episode_count'], 2),
            round(kwargs['total_error'], 2),
            round(kwargs['total_reward'], 2),
            round(avg_reward, 2),
            kwargs['num_steps'],
            episode_duration))

    def on_step_start(self, *args, **kwargs):
        pass

    def on_step_end(self, *args, **kwargs):
        pass

    def on_train_start(self, *args, **kwargs):
        pass

    def on_train_end(self, *args, **kwargs):
        pass

    def get_episode_metrics(self):
        metrics = {}

        if len(self.episode_metrics) == 0: return pd.DataFrame(metrics)

        # Convert from list of dictionaries to dictionary of lists
        for k in self.episode_metrics[0].keys():
            metrics[k] = [d[k] for d in self.episode_metrics]

        df = pd.DataFrame(metrics)
        if 'total_reward' in df.columns:
            df['mean_reward'] = pd.rolling_mean(df['total_reward'], window=50, min_periods=0)

        return df

