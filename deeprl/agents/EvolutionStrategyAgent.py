import numpy as np

from . import AbstractAgent


class EvolutionStrategyAgent(AbstractAgent):
    pass

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _update_weights(self, epsilon):
        pass

    def train(self):
        sigma = 0.1 # Pass as input?  Constructor?
        alpha = 1e-3
        epsilons = []
        rewards = []
        n = 100  # Population size (& # of episodes per training update)
        epochs = 20

        for epoch in range(epochs):

            starting_weights = self.model.get_weights()
            t1 = starting_weights[0] + 1.

            for i in range(n): # # population size
                epsilon = []
                new_weights = []
                for w in starting_weights:
                    e = np.random.randn(*w.shape)
                    new_weights.append(w + sigma * e)
                    epsilon.append(e)
                self.model.set_weights(new_weights)

                self.episode_start(episode_count=epoch)

                # Run episode
                total_reward, total_error, step_count = self._run_episode(0, True)

                # Fire any notifications
                self.episode_end(episode_count=epoch, total_reward=total_reward, total_error=total_error, num_steps=step_count)

                epsilons.append(epsilon)
                rewards.append(total_reward)

            new_weights = []
            for layer in range(len(starting_weights)):
                t0 = [rewards[i] * epsilons[i][layer] for i in range(n)]
                t1 = sum(t0)
                t2 = alpha * 1. / (n * sigma) * t1
                new_weights.append(starting_weights[layer] + t2)

            self.model.set_weights(new_weights)






    def _run_episode(self, total_steps, render):
            episode_done = False
            total_reward = 0
            total_error = 0
            step_count = 0

            s = self.env.reset()  # Get initial state observation

            while not episode_done:
                if render:
                    self.env.render()
                s = np.asarray(s)

                self.step_start(step=step_count, total_steps=total_steps, s=s)

                q_values = self.model.predict_on_batch(s.reshape(1, -1))
                a = self.policy(q_values)

                s_prime, r, episode_done, _ = self.env.step(a)

                step_count += 1
                total_steps += 1
                total_reward += r  # Track rewards without shaping

                s, a, r, s_prime, episode_done = self.preprocess_state(s, a, r, s_prime, episode_done)

                # Force the episode to end if we've reached the maximum number of steps allowed
                if self.max_steps_per_episode and step_count >= self.max_steps_per_episode:
                    episode_done = True

                self.step_end(step=step_count, total_steps=total_steps, s=s, s_prime=s_prime, a=a, r=r,
                              episode_done=episode_done)
                self.logger.debug("S: {}\tA: {}\tR: {}\tS': {}\tDone: {}".format(s, a, r, s_prime, episode_done))

                s = s_prime

            return total_reward, total_error, step_count

            # Sample noise
            # Update weights
            # run episode
            # reverse weight change
            # average returns and





    '''
    train
    run episode
    update weights


    for each episode, sample random noise
    (temp) update weights
    run episode
    get total reward
    compute gradient
    actual weight update

    batch normalization
    '''