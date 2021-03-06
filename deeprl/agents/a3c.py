from . import AbstractAgent
from deeprl.memories import TrajectoryMemory
import gym
import multiprocessing
from multiprocessing import Process
import numpy as np
import keras.backend as K
import keras.models
import pickle

DEFAULT_TRAIN_INTERVAL = 5 # Number of steps between weight updates

def copy_local(arg):
    return pickle.loads(pickle.dumps(arg))

class A3CWorker(AbstractAgent):
    # TODO: Allow shared weights between actor & critic
    # TODO: Support id/name in metrics collection

    def __init__(self, id, shared_config, shared_actor, shared_critic, **kwargs):
        assert 'env' in shared_config, 'Configuration missing key "env".'

        self.id = id
        self.name = multiprocessing.current_process().name
        self.config = shared_config

        self.local_actor = self._build_model(shared_actor, lambda y_true, y_pred: y_pred)
        self.local_critic = self._build_model(shared_critic, 'mse')
        self.shared_actor = shared_actor
        self.shared_critic = shared_critic

        self.train_actor_on_batch = self._build_actor_train_func(self.local_actor)
        self.train_interval = self.config.get('train_every_n', DEFAULT_TRAIN_INTERVAL)

        env = gym.make(self.config['env'])
        memory = TrajectoryMemory(maxlen=self.train_interval)
        max_steps = self.config['max_steps_per_episode']

        super(A3CWorker, self).__init__(env, gamma=self.config['gamma'], memory=memory, max_steps_per_episode=max_steps, **kwargs)

        # Construct an upper triangular matrix where each diagonal 0..k = the discount factor raised to that power
        # Once constructed, the n-step return can be computed as Gr where G is the matrix of discount factors
        # and r is the vector of rewards observed at each time step
        self.G = sum([np.diagflat(np.ones(self.train_interval - i) * self.gamma ** i, k=i) for i in range(self.train_interval)])

        # Automatically refresh the local weights when each episode starts
        self.episode_start += self._refresh_local_weights

        self.logger.info(f'A3CWorker instance {self.name} created.')

    def _refresh_local_weights(self, *args, **kwargs):
        # Refresh the weights of the local models from the shared model weights
        self.orig_actor_weights = copy_local(self.shared_actor.get_weights())
        self.orig_critic_weights = copy_local(self.shared_critic.get_weights())

        self.local_actor.set_weights(self.orig_actor_weights)
        self.local_critic.set_weights(self.orig_critic_weights)

        self.logger.debug('Refreshed local weights from shared model.')


    def _update_weights(self):
        # Weight updates are only done at the end of an episode or after a specific number of steps have been taken.
        # Abort training if episode isn't done and the current step doesn't fall on the training schedule.
        if not self._status.episode_done and self._status.step % self.train_interval != 0:
            return

        states, actions, rewards, s_primes, flags = self.memory.sample()
        batch_size = states.shape[0]

        # Total n-step return from each state
        R = np.dot(self.G, rewards)

        # If final state was terminate than it's value is 0.
        # Otherwise, we must include the discounted value of the last state observed.
        # Discount factor for V(s') is gamma * discount for last reward observed before s'.
        if flags[-1] == False:
            terminal_val = self.local_critic.predict(s_primes[-1, :].reshape((1, -1)))
            R += (self.G[:, -1] * self.gamma * terminal_val).reshape(R.shape)

        # Train the actor network
        mask = np.zeros((batch_size, self.num_actions))
        mask[range(batch_size), actions.astype('int32').ravel()] = 1
        V = self.local_critic.predict_on_batch(states)
        advantage = R - V
        actor_err = self.train_actor_on_batch([states, mask, advantage])

        # Train the critic network
        critic_err = self.local_critic.train_on_batch(states[:-1, :], R[:-1])

        # Update the shared model weights and refresh local weights
        actor_delta = [new - old for new, old in zip(self.local_actor.get_weights(), self.orig_actor_weights)]
        self.orig_actor_weights = copy_local(self.shared_actor.add_delta(actor_delta))
        self.local_actor.set_weights(self.orig_actor_weights)

        critic_delta = [new - old for new, old in zip(self.local_critic.get_weights(), self.orig_critic_weights)]
        self.orig_critic_weights = copy_local(self.shared_critic.add_delta(critic_delta))  # Add deltas and get updated weights
        self.local_critic.set_weights(self.orig_critic_weights)  # Update the model with new weights

        self.logger.debug('Shared weights updated.')


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
        # Actor network returns probability of choosing each action in current state
        actions = self.local_actor.predict_on_batch(state)

        return np.random.choice(np.arange(actions.size), p=actions.ravel())



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

    worker = A3CWorker(id, c.root.config, c.root.config['actor'], c.root.config['critic'], logger=c.root.logger)
    worker.train(render_every_n=c.root.config.get('render_every_n', 0) if id == 0 else 0)    # Only root worker should render)

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


class A3CAgent(AbstractAgent):
    """
    And implementation of Google DeepMind's Asynchronous Advantage Actor-Critic algorithm.
    Uses an actor network to ouput the probability of taking each (discrete) action.
    Uses a critic network to determine the q-value of the action taken.


    Asynchronous Methods for Deep Reinforcement Learning (https://arxiv.org/pdf/1602.01783.pdf)
    """
    def __init__(self, actor, critic, controller=None, **kwargs):

        super(A3CAgent, self).__init__(**kwargs)

        if 'policy' in kwargs:
            self.logger.warn('`Policy` parameter is invalid for A3C agents and will be ignored.')

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

        self.controller.config['gamma'] = self.gamma


    def train(self, max_episodes, num_workers, **kwargs):
        """Train the agent in the environment for a specified number of episodes."""

        self.controller.config['env'] = self.env.spec.id
        self.controller.config['max_episodes'] = max_episodes

        self.controller.config.update(kwargs)

        self.controller.config['max_steps_per_episode'] = self.max_steps_per_episode
#        self.controller.config['render_every_n'] = kwargs.get('render_every_n', 0)
        #self.controller.config['train_every_n']

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
                self.logger.info(f'Terminating {p}.')
                p.terminate()
                p.join()


