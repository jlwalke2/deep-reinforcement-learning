import logging
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class AbstractPolicy():
    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class BoltzmannPolicy(AbstractPolicy):
    def __init__(self, min_temp=0.0, max_temp=100):
        super().__init__()

        self.min_temp = min_temp
        self.max_temp = max_temp

    def __call__(self, qvalues, *args, **kwargs):
        # A simple implementation of Boltzmann can have issues with large q-values since the exponentiation
        # can result in overflow, leading to an incorrect policy.  To get around this we scale the q-values down
        # into a reasonable range prior to exponentiation.  Since the probabilities only depend on the *difference*
        # between q-values instead of the actual values, we can +/- q-values arbitrarily without affecting the outcome.
        # Additionally, given two numbers a, b where a = b + d for some delta d, the probability of choosing b rapidly
        # approaches 0 as d increases.  Given this, we need only consider q-values that are within d of the max q-value.
        delta = 20
        ignore = qvalues < qvalues.max() - delta
        keep = np.logical_not(ignore)

        adj_q = qvalues - qvalues[keep].min()
        if np.any(ignore):
            adj_q[ignore] = 0               # These are now negative, so zero out to avoid underflow

        probs = np.exp(adj_q)
        if np.any(ignore):
            probs[ignore] = 0               # exp(0) = 1 so zero out these probabilities

        probs /= np.sum(probs)              # Normalize remaining probabilities to sum to 1
        np.clip(probs, 0., 1., out=probs)   # Ensure no probability outside 0..1
        probs = probs.ravel()

        logger.debug('Q={}\tP={}'.format(qvalues, probs))
        return np.random.choice(range(qvalues.size), p=probs)


class EpsilonGreedyPolicy(AbstractPolicy):
    def __init__(self, max=1.0, min=0.0, decay=0.9, exploration_episodes=0):
        super().__init__()

        self.epsilon = max
        self.min = min
        self.decay = decay
        self.exploration_episodes = exploration_episodes

    def __call__(self, qvalues, *args, **kwargs):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(range(qvalues.size))
        else:
            # Get the index (action #s) of the actions with max Q Values
            best_actions = np.argwhere(qvalues == np.max(qvalues))[:, 1]

            # Randomly choose one of the best actions
            a = np.random.choice(best_actions.flatten())
            logger.debug('Epsilon: {}  Q: {}'.format(self.epsilon, qvalues))
            return a

    def on_episode_end(self, *args, **kwargs):
        assert 'episode' in kwargs.keys()

        if kwargs['episode'] > self.exploration_episodes:
            self.epsilon = max(self.epsilon * self.decay, self.min)

        logger.info('Epsilon: {}'.format(round(self.epsilon, 2)))


class GreedyPolicy(EpsilonGreedyPolicy):
    '''
    A pure greedy policy that always chooses the action with the highest q-value.
    '''

    def __init__(self):
        super().__init__(min=0, max=0, decay=0, exploration_episodes=0)
