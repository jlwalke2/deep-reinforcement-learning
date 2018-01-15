from shutil import rmtree
from tempfile import mkdtemp

import gym
import gym.wrappers

from .AbstractAgent import AbstractAgent


class DeepQAgent(AbstractAgent):
    def __init__(self, model, gamma=0.9, *args, **kwargs):
        super(DeepQAgent, self).__init__(*args, **kwargs)
        self.model = model
        self.gamma = gamma
        self.preprocess_steps = []


    def train(self, steps_before_training=None, **kwargs):
        # Unless otherwise specified, assume training doesn't start until a full sample of steps is observed
        self._steps_before_training = steps_before_training or self.memory.sample_size

        super(DeepQAgent, self).train(**kwargs)


    # def train(self, max_episodes=500, steps_before_training=None, target_model_update=1000, render_every_n=1, upload=False):
    #     # Unless otherwise specified, assume training doesn't start until a full sample of steps is observed
    #     if steps_before_training is None:
    #         steps_before_training = self.memory.sample_size
    #
    #     from collections import deque
    #     recent_rewards = deque(maxlen=50)
    #     try:
    #         total_steps = 0
    #
    #         for episode_count in range(1, max_episodes + 1):
    #             self.episode_start(episode_count=episode_count)
    #
    #             render = episode_count % render_every_n == 0
    #             total_reward, total_error, steps_in_episode = self._run_episode(target_model_update, steps_before_training, total_steps, render)
    #
    #             # Fire any notifications
    #             self.episode_end(episode_count=episode_count, total_reward=total_reward, total_error=total_error, num_steps=steps_in_episode)
    #             recent_rewards.append(total_reward)
    #             avg_reward = sum(recent_rewards) / len(recent_rewards)
    #             self.logger.info('Episode {} \tError: {} \tReward: {} \tAvg Reward: {}'.format(episode_count, total_error, total_reward, avg_reward))
    #             total_steps += steps_in_episode
    #
    #         self.env.close()
    #
    #     except KeyboardInterrupt:
    #         return
