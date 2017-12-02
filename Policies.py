import logging
import numpy as np

class AbstractPolicy():
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class BoltzmannPolicy(AbstractPolicy):
    def __init__(self, min_temp=0.0, max_temp=100):
        super().__init__()

        self.min_temp = min_temp
        self.max_temp = max_temp

    def __call__(self, qvalues, *args, **kwargs):
        # TODO: Better way to handle overflow?

        probs = np.round(np.exp(qvalues.astype('float64')), 5)  # Convert to float64 to avoid overflow from exp
        probs /= np.sum(probs)              # Normalize to sum to 1
        probs[0] -= (np.sum(probs) - 1.0)   # Total probability can be close but != 1.0.  +/- any difference arbitrarily to the first action
        np.clip(probs, 0., 1., out=probs)   # Ensure no probability outside 0..1
        probs = probs.ravel()

        self.logger.debug('Q={}\tP={}'.format(qvalues, probs))
        return np.random.choice(range(qvalues.size), p=probs)


class EpsilonGreedyPolicy(AbstractPolicy):
    def __init__(self, max=1.0, min=0.0, decay=0.9, exploration_episodes=0):
        super().__init__()

        self.epsilon = max
        self.min = min
        self.decay = decay
        self.exploration_episodes = exploration_episodes
        self.episode_count = 0

    def __call__(self, qvalues, *args, **kwargs):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(range(qvalues.size))
        else:
            # Get the index (action #s) of the actions with max Q Values
            best_actions = np.argwhere(qvalues == np.max(qvalues))[:, 1]

            # Randomly choose one of the best actions
            a = np.random.choice(best_actions.flatten())
            self.logger.debug('Epsilon: {}  Q: {}'.format(self.epsilon, qvalues))
            return a

    def on_episode_complete(self):
        self.episode_count += 1
        if self.episode_count > self.exploration_episodes:
            self.epsilon = max(self.epsilon * self.decay, self.min)

        self.logger.info('Episode: {}  Epsilon: {}'.format(self.episode_count, round(self.epsilon, 2)))
