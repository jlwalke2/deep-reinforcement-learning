from .AbstractAgent import AbstractAgent
from ..utils.async import ModelManager
from Memories import TrajectoryMemory
import gym
import multiprocessing
from multiprocessing import Process
import numpy as np
import keras.backend as K
import keras.models

def run_worker(id, config, actor, critic):
    worker = A3CWorker(id, config, actor, critic)
    worker.train()


class A3CWorker():
    # TODO: Implement shared logger
    # TODO: Pass gamma in config
    # TODO: Allow shared weights between actor & critic
    # TODO: Support id/name in metrics collection

    def __init__(self, id, shared_config, shared_actor, shared_critic):
        assert 'env' in shared_config, 'Configuration missing key "env".'

        self.id = id
        self.name = multiprocessing.current_process().name
        self.config = shared_config
        self.env = gym.make(self.config['env'])
        self.local_actor = self._build_model(shared_actor, lambda y_true, y_pred: y_pred)
        self.local_critic = self._build_model(shared_critic, 'mse')
        self.shared_actor = shared_actor
        self.shared_critic = shared_critic
        self.num_actions = self.env.action_space.n
        self.train_actor_on_batch = self._build_actor_train_func(self.local_actor)

        print(self.name)


    def _build_actor_train_func(self, model):
        if isinstance(model, keras.models.Sequential):
            model = model.model  # Unwrap internal Model from Sequential object

        # Define a custom objective function to maximize
        def objective(action, mask, advantage):
            # Since Keras optimizers always minimize "loss" we'll have to minimize the negative of the objective
            return -(K.sum(K.log(action + 1e-30) * mask, axis=-1) * advantage)

        mask = K.placeholder(shape=model.output.shape, name='mask')
        advantage = K.placeholder(shape=(None,1), name='advantage')
        loss = objective(model.output, mask, advantage) # Compute objective/loss tensor

        # Build a Keras function to run in the inputs through the model, return the outputs, and perform the
        # (weight) updates created by the optimizer
        return K.function(inputs=model._feed_inputs + [mask, advantage],
                   outputs=[loss],
                   updates=model.optimizer.get_updates(params=model._collected_trainable_weights, loss=loss))

    def _build_model(self, shared_model, loss):
        t, c = shared_model.get_model()
        local_model = t.from_config(c)

        t, c = shared_model.get_optimizer()
        optimizer = t.from_config(c)

        local_model.compile(optimizer, loss=loss)
        # input_layers
        # layers
        # output
        from keras.utils.vis_utils import plot_model
        plot_model(local_model, to_file='model.png')
        return local_model

    def choose_action(self, state):
        # Actor network returns probability of choosing each action in current state
        actions = self.local_actor.predict_on_batch(state)

        # Select the action to take
        return np.random.choice(np.arange(actions.size), p=actions.ravel())

    def train(self):
        max_episodes = self.config['max_episodes']
        train_interval = 5
        max_steps = self.config['max_steps_per_episode']
        gamma = self.config['gamma']
        memory = TrajectoryMemory(maxlen=train_interval)

        # Construct an upper triangular matrix where each diagonal 0..k = the discount factor raised to that power
        # Once constructed, the n-step return can be computed as Gr where G is the matrix of discount factors
        # and r is the vector of rewards observed at each time step
        G = sum([np.diagflat(np.ones(train_interval - i) * gamma ** i, k=i) for i in range(train_interval)])

        for episode in range(max_episodes):
            orig_actor_weights = self.shared_actor.get_weights()  # Refresh local copies of weights
            orig_critic_weights = self.shared_critic.get_weights()
            self.local_actor.set_weights(orig_actor_weights)
            self.local_critic.set_weights(orig_critic_weights)

            done = False
            step = 0
            s = self.env.reset()

            while not done:
                if self.id == 0 and episode % 25 == 0:
                    self.env.render()
                step += 1
                a = self.choose_action(s.reshape(1, -1))
                s_prime, r, done, _ = self.env.step(a)

                # Store experience
                memory.append((s, a, r, s_prime, done))
                s = s_prime

                # Terminate loop anyways if max steps reached.
                # Episode not marked as terminal so V(s') estimate will be used
                if step == max_steps:
                    done = True

                if done or step % train_interval == 0:
                    states, actions, rewards, s_primes, flags = memory.sample()
                    batch_size = states.shape[0]

                    # Total n-step return from each state
                    R = np.dot(G, rewards)

                    # If final state was terminate than it's value is 0.
                    # Otherwise, we must include the discounted value of the last state observed.
                    # Discount factor for V(s') is gamma * discount for last reward observed before s'.
                    if flags[-1] == False:
                        terminal_val = self.local_critic.predict(s_primes[-1, :].reshape((1, -1)))
                        R += (G[:, -1] * gamma * terminal_val).reshape(R.shape)


                    # Train the actor network
                    mask = np.zeros((batch_size, self.num_actions))
                    mask[range(batch_size), actions.astype('int32').ravel()] = 1
                    V = self.local_critic.predict_on_batch(states)
                    advantage = R - V
                    actor_err = self.train_actor_on_batch([states, mask, advantage])

                    # Train the critic network
                    critic_err = self.local_critic.train_on_batch(states[:-1, :], R[:-1])

                    print('{} | episode = {},  step = {},  actor error = {},  critic error = {}'.format(self.name, episode, step, '', critic_err))

                    # Update the shared model weights and refresh local weights
                    actor_delta = [new - old for new, old in zip(self.local_actor.get_weights(), orig_actor_weights)]
                    orig_actor_weights = self.shared_actor.add_delta(actor_delta)
                    self.local_actor.set_weights(orig_actor_weights)

                    critic_delta = [new - old for new, old in zip(self.local_critic.get_weights(), orig_critic_weights)]
                    orig_critic_weights = self.shared_critic.add_delta(critic_delta)  # Add deltas and get updated weights
                    self.local_critic.set_weights(orig_critic_weights)  # Update the model with new weights

        pass





# A3C https://arxiv.org/pdf/1602.01783.pdf

class A3CAgent(AbstractAgent):
    def __init__(self, actor, critic, **kwargs):

        kwargs['model'] = actor
        super(A3CAgent, self).__init__(**kwargs)

        self.actor = actor
        self.critic = critic

    def _build_shared_model(self, model, manager):

        return manager.Model(model_type=type(model),
                                   model_config=model.get_config(),
                                   optimizer_type=type(model.optimizer),
                                   optimizer_config=model.optimizer.get_config(),
                                   weights=model.get_weights())


    def train(self, max_episodes, num_threads, **kwargs):
        """Train the agent in the environment for a specified number of episodes."""
        manager = ModelManager()
        manager.start()

        actor = self._build_shared_model(self.actor, manager)
        critic = self._build_shared_model(self.critic, manager)

        config = manager.dict()
        config['env'] = self.env.spec.id
        config['gamma'] = 0.99 # TODO: pass in constructor
        config['max_episodes'] = max_episodes
        config['max_steps_per_episode'] = self.max_steps_per_episode

        processes = [Process(target=run_worker, args=(i, config, actor, critic), name='Worker %s' % i) for i in range(num_threads)]

        for p in processes:
            p.start()

        for p in processes:
            p.join()
