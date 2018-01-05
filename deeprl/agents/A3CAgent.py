from .AbstractAgent import AbstractAgent
from ..utils.async import ModelManager
from Memories import TrajectoryMemory
import gym
import multiprocessing
from multiprocessing import Process
import numpy as np
import keras.backend as K
import keras.models

def run_worker(config, actor, critic):
    worker = A3CWorker(config, actor, critic)
    worker.train()


class A3CWorker():
    # TODO: Implement shared logger
    # TODO: Pass gamma in config

    def __init__(self, shared_config, shared_actor, shared_critic):
        assert 'env' in shared_config, 'Configuration missing key "env".'

        self.config = shared_config
        self.env = gym.make(self.config['env'])
        self.local_actor = self._build_model(shared_actor, lambda y_true, y_pred: y_pred)
        self.local_critic = self._build_model(shared_critic, 'mse')
        self.shared_actor = shared_actor
        self.shared_critic = shared_critic

        self.local_actor.train_on_batch = self._build_train_func(self._get_train_function(self.local_actor))

        print(multiprocessing.current_process().name)


    def _build_train_func(self, f):
        '''
        # Define loss tensor?
        # Define weight tensors?
        # Compute gradient of loss wrt weights
        # Name scopes?

        log(a | s) * (R - V)
        # Inputs:  states, action mask, returns, values
        # Outputs: loss?
        # Updates: gradient

        Compute objective function
        Compute gradient of weights wrt objective
        update weights
        '''

        def wrapper(y_true, y_pred, mask, advantage):
            # Zero out values for all actions except the one taken
            y_pred = K.print_tensor(y_pred * mask, 'Masked actions')
            t1 = K.log(y_pred)
            t2 = t1 * advantage

            # log(action) * (R-V)
            loss = f(y_true, t2)    # Assumes loss function just returns y_pred as the loss
            return loss
        return wrapper


    def _build_model(self, shared_model, loss):
        t, c = shared_model.get_model()
        local_model = t.from_config(c)

        t, c = shared_model.get_optimizer()
        optimizer = t.from_config(c)

        local_model.compile(optimizer, loss=loss)
        return local_model

    def _get_train_function(self, model):
        if isinstance(model, keras.models.Sequential):
            model = model.model # Sequential wraps an internal Model object

        # Force Keras to construct its standard training function
        model._make_train_function()

        # Return the function
        return model.train_function


    def choose_action(self, state):
        # Actor network returns probability of choosing each action in current state
        actions = self.local_actor.predict_on_batch(state)

        # Select the action to take
        return np.random.choice(np.arange(actions.size), p=actions.ravel())

    def train(self):
        max_episodes = 5
        train_interval = 5
        max_steps = 50
        gamma = 0.9
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

            for episode in range(max_episodes):
                done = False
                step = 0
                s = self.env.reset()

                while not done:
                    step += 1
                    a = self.choose_action(s)
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

                        # Total n-step return from each state
                        R = np.dot(G, rewards)

                        # If final state was terminate than it's value is 0.
                        # Otherwise, we must include the discounted value of the last state observed.
                        # Discount factor for V(s') is gamma * discount for last reward observed before s'.
                        if flags[-1] == False:
                            terminal_val = self.local_critic.predict(s_primes[-1, :].reshape((1, -1)))
                            R += (G[:, -1] * gamma * terminal_val).reshape(R.shape)

#                        actor_err = self.local_actor.
                        critic_err = self.local_critic.train_on_batch(states[:-1, :], R[:-1])


                        print('{} | step = {},  error = {}'.format(multiprocessing.current_process().name, step, critic_err))
                        critic_delta = [new - old for new, old in zip(self.local_critic.get_weights(), orig_critic_weights)]
                        orig_critic_weights = self.shared_critic.add_delta(critic_delta)  # Add deltas and get updated weights
                        self.local_critic.set_weights(orig_critic_weights)  # Update the model with new weights
                        # TODO: Send deltas

            # TODO: Update weight diffs
            pass
            # compute gradient
            # update dict







# A3C https://arxiv.org/pdf/1602.01783.pdf

class A3CAgent(AbstractAgent):
    def __init__(self, actor, critic, **kwargs):

        kwargs['model'] = actor
        super(A3CAgent, self).__init__(**kwargs)

        self.actor = actor
        self.critic = critic
        pass
        # Memory??
        # setup shared weights

    def _build_shared_model(self, model, manager):

        return manager.Model(model_type=type(model),
                                   model_config=model.get_config(),
                                   optimizer_type=type(model.optimizer),
                                   optimizer_config=model.optimizer.get_config(),
                                   weights=model.get_weights())


    def train(self, max_steps, num_threads, **kwargs):
        """Train the agent in the environment for a specified number of episodes."""
        # self._target_model_update = target_model_update
        # self.step_end += self._update_model_weights
        # self.step_end += self._update_target_weights
        #
        # # Run the training loop
        # super().train(**kwargs)

        #manager = ModelManager()
        #manager.start()

        manager = ModelManager()
        manager.start()

        actor = self._build_shared_model(self.actor, manager)
        critic = self._build_shared_model(self.critic, manager)

        config = manager.dict()
        config['env'] = self.env.spec.id
        config['gamma'] = 0.99 # TODO: pass in constructor

        processes = [Process(target=run_worker, args=(config, actor, critic)) for _ in range(num_threads)]

        for p in processes:
            p.start()

        for p in processes:
            p.join()

        # Copy env
        # Initialize Memory
        # Copy weights
        # Run episode,


'''
Create jobs w/: env spec & model spec(s) & logger?
Each job:
    create env
    create memory
    duplicate model(s)
    request weights
    until done or terminated:
        run episode
        compute weight updates
        send updates
        get new weights

'''
# Weight Handler
# Manager class
