from .AbstractAgent import AbstractAgent
from ..utils.async import ModelManager
from Memories import TrajectoryMemory
import gym
import multiprocessing
from multiprocessing import Process
import numpy as np

def worker(config, actor, critic):
    def build_model(shared_model, loss):
        t, c = shared_model.get_model()
        local_model = t.from_config(c)

        t, c = shared_model.get_optimizer()
        optimizer = t.from_config(c)

        local_model.compile(optimizer, loss=loss)

        return local_model

    env = gym.make(config['env'])
    # TODO: pass loss function
    # TODO: Implement shared logger
    # TODO: Pass gamma
    print(multiprocessing.current_process().name)

    local_actor = build_model(actor, 'mse')
    local_critic = build_model(critic, 'mse')

    # TODO: pull from config
    max_episodes = 5
    train_interval = 5
    max_steps = 50
    gamma = 0.9
    memory = TrajectoryMemory(maxlen=train_interval)

    # Construct an upper triangular matrix where each diagonal 0..k = the discount factor raised to that power
    # Once constructed, the n-step return can be computed as Gr where G is the matrix of discount factors
    # and r is the vector of rewards observed at each time step
    G = sum([np.diagflat(np.ones(train_interval - i) * gamma**i, k=i) for i in range(train_interval)])


    for episode in range(max_episodes):
        orig_actor_weights = actor.get_weights() # Refresh local copies of weights
        orig_critic_weights = critic.get_weights()
        local_actor.set_weights(orig_actor_weights)
        local_critic.set_weights(orig_critic_weights)

        for episode in range(max_episodes):
            done = False
            step = 0
            s = env.reset()

            while not done:
                # TODO: select action
                step += 1
                a = 0
                s_prime, r, done, _ = env.step(a)

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
                        terminal_val = local_critic.predict(s_primes[-1, :].reshape((1, -1)))
                        R += (G[:,-1] * gamma * terminal_val).reshape(R.shape)

                    error = local_critic.train_on_batch(states[:-1, :], R[:-1])
                    print('{} | step = {},  error = {}'.format(multiprocessing.current_process().name, step, error))
                    critic_delta = [new - old for new, old in zip(local_critic.get_weights(), orig_critic_weights)]
                    orig_critic_weights = critic.add_delta(critic_delta) # Add deltas and get updated weights
                    local_critic.set_weights(orig_critic_weights)        # Update the model with new weights
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

        processes = [Process(target=worker, args=(config, actor, critic)) for _ in range(num_threads)]

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
