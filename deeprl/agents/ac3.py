from .abstract import AbstractAgent
from deeprl.memories import TrajectoryMemory
import gym
import multiprocessing
from multiprocessing import Process
import numpy as np
import keras.backend as K
import keras.models
import pickle
import random

import logging
#logger = logging.getLogger(__name__)
logger = multiprocessing.log_to_stderr()

def copy_local(arg):
    return pickle.loads(pickle.dumps(arg))

class A3CWorker(AbstractAgent):
    # TODO: Implement shared logger
    # TODO: Pass gamma in config
    # TODO: Allow shared weights between actor & critic
    # TODO: Support id/name in metrics collection

    def __init__(self, id, shared_config, shared_actor, shared_critic):
        assert 'env' in shared_config, 'Configuration missing key "env".'

        self.id = id
        self.name = multiprocessing.current_process().name
        self.config = shared_config

        self.local_actor = self._build_model(shared_actor, lambda y_true, y_pred: y_pred)
        self.local_critic = self._build_model(shared_critic, 'mse')
        self.shared_actor = shared_actor
        self.shared_critic = shared_critic

        self.train_actor_on_batch = self._build_actor_train_func(self.local_actor)
        self.gamma = self.config['gamma']
        self.train_interval = 5                 # TODO: Pull from config or default

        env = gym.make(self.config['env'])
        memory = TrajectoryMemory(maxlen=self.train_interval)
        max_steps = self.config['max_steps_per_episode']
        logger = multiprocessing.log_to_stderr()

        super(A3CWorker, self).__init__(env, memory=memory, max_steps_per_episode=max_steps)

        logger.info('Starting {}...'.format(self.name))


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
                   updates=model.optimizer.get_updates(params=model._collected_trainable_weights, loss=loss, constraints=[]))

    def _build_model(self, shared_model, loss):
        import importlib

        def load(module_name, class_name, config):
            module = importlib.import_module(module_name)
            class_ = getattr(module, class_name)
            return class_.from_config(c)

        t, c = shared_model.get_model()
        local_model = load(*t, c)

        t, c = shared_model.get_optimizer()
        optimizer = load(*t, c)

        local_model.compile(optimizer, loss=loss)
        # input_layers
        # layers
        # output
        # from keras.utils.vis_utils import plot_model
        # plot_model(local_model, to_file='model.png')
        return local_model

    def choose_action(self, state):
        # TODO: Enable swappable policies
        
        # Actor network returns probability of choosing each action in current state
        actions = self.local_actor.predict_on_batch(state)

        if random.random() < 0.1:
            return np.random.choice(np.arange(actions.size))
        else:
            # Select the action to take
            return np.random.choice(np.arange(actions.size), p=actions.ravel())

    def train(self):
        logger.info('Training started...')
        max_episodes = self.config['max_episodes']

        # Construct an upper triangular matrix where each diagonal 0..k = the discount factor raised to that power
        # Once constructed, the n-step return can be computed as Gr where G is the matrix of discount factors
        # and r is the vector of rewards observed at each time step
        G = sum([np.diagflat(np.ones(self.train_interval - i) * self.gamma ** i, k=i) for i in range(self.train_interval)])
        logger.info(f'Max Episodes = {max_episodes}')

        for episode in range(max_episodes):
            orig_actor_weights = self.shared_actor.get_weights()  # Refresh local copies of weights
            orig_critic_weights = self.shared_critic.get_weights()

            # Pickle weights to ensure local copies are being used, not netrefs
            orig_actor_weights = copy_local(orig_actor_weights)
            orig_critic_weights = copy_local(orig_critic_weights)
            self.local_actor.set_weights(orig_actor_weights)
            self.local_critic.set_weights(orig_critic_weights)

            done = False
            step = 0
            s = self.env.reset()
            logger.info(f'Starting episode {episode}')
            total_reward = 0

            while not done:
                if self.id == 0 and episode % 25 == 0:  # TODO: Pull from config
                    self.env.render()
                step += 1
                a = self.choose_action(s.reshape(1, -1))
                s_prime, r, done, _ = self.env.step(a)

                # Store experience
                self.memory.append((s, a, r, s_prime, done))
                s = s_prime
                total_reward += r

                # Terminate loop anyways if max steps reached.
                # Episode not marked as terminal so V(s') estimate will be used
                if step == self.max_steps_per_episode:
                    done = True

                if done or step % self.train_interval == 0:
                    states, actions, rewards, s_primes, flags = self.memory.sample()
                    batch_size = states.shape[0]

                    # Total n-step return from each state
                    R = np.dot(G, rewards)

                    # If final state was terminate than it's value is 0.
                    # Otherwise, we must include the discounted value of the last state observed.
                    # Discount factor for V(s') is gamma * discount for last reward observed before s'.
                    if flags[-1] == False:
                        terminal_val = self.local_critic.predict(s_primes[-1, :].reshape((1, -1)))
                        R += (G[:, -1] * self.gamma * terminal_val).reshape(R.shape)

                    # Train the actor network
                    mask = np.zeros((batch_size, self.num_actions))
                    mask[range(batch_size), actions.astype('int32').ravel()] = 1
                    V = self.local_critic.predict_on_batch(states)
                    advantage = R - V
                    actor_err = self.train_actor_on_batch([states, mask, advantage])

                    # Train the critic network
                    critic_err = self.local_critic.train_on_batch(states[:-1, :], R[:-1])

                    logger.info('{} | episode = {},  step = {},  actor error = {},  critic error = {}'.format(self.name, episode, step, '', critic_err))

                    # Update the shared model weights and refresh local weights
                    actor_delta = [new - old for new, old in zip(self.local_actor.get_weights(), orig_actor_weights)]
                    orig_actor_weights = copy_local(self.shared_actor.add_delta(actor_delta))
                    self.local_actor.set_weights(orig_actor_weights)

                    critic_delta = [new - old for new, old in zip(self.local_critic.get_weights(), orig_critic_weights)]
                    orig_critic_weights = copy_local(self.shared_critic.add_delta(critic_delta))  # Add deltas and get updated weights
                    self.local_critic.set_weights(orig_critic_weights)  # Update the model with new weights


def start_worker(id):
    import rpyc, time, socket

    for i in range(60):
        try:
            c = rpyc.connect(socket.gethostname(), port=34392,
                             config=dict(allow_pickle=True,
                                         allow_public_attrs=True,
                                         allow_all_attrs=True,
                                         allow_setattr=True,
                                         allow_getattr=True))
            break
        except ConnectionRefusedError:
            time.sleep(1)

    if 'c' not in locals():
        raise Exception()

    worker = A3CWorker(id, c.root.config, c.root.config['actor'], c.root.config['critic'])
    worker.train()

def start_registry():
    from rpyc.utils.registry import UDPRegistryServer, REGISTRY_PORT, DEFAULT_PRUNING_TIMEOUT
    registry = UDPRegistryServer(host='0.0.0.0', port=REGISTRY_PORT, pruning_timeout=DEFAULT_PRUNING_TIMEOUT)
    registry.start()

def start_service():
    from ..utils.async import SharedModelService
    import rpyc

    s = rpyc.utils.server.ThreadedServer(SharedModelService(), port=34392,
                                         auto_register=True)
    s.start()

# A3C https://arxiv.org/pdf/1602.01783.pdf

class A3CAgent(AbstractAgent):
    def __init__(self, actor, critic, controller=None, **kwargs):

        super(A3CAgent, self).__init__(**kwargs)

        self.actor = actor
        self.critic = critic
        self._non_worker_processes = []

        if controller is None:
            # NOTE:  Following result if .discover called and no registry service is found
            # rpyc.utils.factory.DiscoveryError: no servers exposing 'sharedmodel' were found
            import rpyc, time, socket

            p1 = Process(target=start_registry, name='RPyC Service', daemon=True)
            p1.start()
            self._non_worker_processes.append(p1)
            time.sleep(5)

            p2 = Process(target=start_service, name='RPyC Service', daemon=True)
            p2.start()
            self._non_worker_processes.append(p2)
            time.sleep(5)

            for i in range(60):
                try:
                    c = rpyc.connect(socket.gethostname(), port=34392,
                                     config={"allow_pickle": True, "allow_public_attrs": True, "allow_all_attrs": True})
                    break
                except ConnectionRefusedError:
                    time.sleep(1)

            if 'c' not in locals():
                raise Exception()

            self.controller = c.root
        else:
            self.controller = controller


    def train(self, max_episodes, num_workers, **kwargs):
        """Train the agent in the environment for a specified number of episodes."""

        from ..utils import async

        self.controller.config['env'] = self.env.spec.id
        self.controller.config['gamma'] = 0.99 # TODO: pass in constructor
        self.controller.config['max_episodes'] = max_episodes
        self.controller.config['max_steps_per_episode'] = self.max_steps_per_episode

        self.controller.create_model(name='actor',
                                     model_type=(type(self.actor).__module__, type(self.actor).__name__),
                                     model_config=self.actor.get_config(),
                                     optimizer_type=(type(self.actor.optimizer).__module__, type(self.actor.optimizer).__name__),
                                     optimizer_config=self.actor.optimizer.get_config(),
                                     weights=self.actor.get_weights())

        self.controller.create_model(name='critic',
                                     model_type=(type(self.critic).__module__, type(self.critic).__name__),
                                     model_config=self.critic.get_config(),
                                     optimizer_type=(
                                     type(self.critic.optimizer).__module__, type(self.critic.optimizer).__name__),
                                     optimizer_config=self.critic.optimizer.get_config(),
                                     weights=self.critic.get_weights())

        try:
            processes = [Process(target=start_worker, args=(i,), name='Worker %s' % i) for i in range(num_workers)]

            for p in processes:
                p.start()

            for p in processes:
                p.join()
        finally:
            for p in self._non_worker_processes:
                logger.info(f'Terminating {p}.')
                p.terminate()
                p.join()


