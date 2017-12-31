import keras.backend as K
from keras.models import Model, Sequential, Input
from .AbstractAgent import AbstractAgent
import gym
import multiprocessing
from multiprocessing.managers import SyncManager
from multiprocessing import Process
import logging
from deeprl.utils.Monitor import Monitor


def worker(d):

    env = gym.make(d['env'])
    # TODO: Set optimizer & compile
    # TODO: Implement shared logger
    print(multiprocessing.current_process().name)
    actor = d['actor_type'].from_config(d['actor_config'])
    actor.set_weights(d['actor_weights'])

    critic = d['critic_type'].from_config(d['critic_config'])
    critic.set_weights(d['critic_weights'])

    # TODO: Create memory
    # TODO: pull from config
    max_episodes = 5
    max_steps = 50
    for episode in range(max_episodes):
        done = False
        step = 1
        s = env.reset()
        while not done:
            # select action
            a = 0
            s, r, s_prime, done = env.step(a)
            step += 1
            if step > max_steps:
                done = True

            # Store experience
        # compute gradient
        # update dict




class SharedModel(object):
    def __init__(self, model_type, model_config, **kwargs):
        super(SharedModel, self).__init__(**kwargs)

        assert 'from_config' in dir(model_type), 'Method from_config() not found on for {}'.format(model_type)

        self.model = model_type.from_config(model_config)
        self.model.compile('sgd', 'mse')

        pass

    def optimizer(self):
        return self.model.optimizer.get_config()

    def get_weights(self):
        return self.model.get_weights()

    def add_deltas(self, d):
        # params = self.weights
        # weight_value_tuples = []
        # param_values = K.batch_get_value(params)
        # for pv, p, w in zip(param_values, params, weights):
        #     if pv.shape != w.shape:
        #         raise ValueError('Optimizer weight shape ' +
        #                          str(pv.shape) +
        #                          ' not compatible with '
        #                          'provided weight shape ' + str(w.shape))
        #     weight_value_tuples.append((p, w))
        # K.batch_set_value(weight_value_tuples)

        print(d)

class ModelManager(SyncManager): pass
ModelManager.register('Model', SharedModel)


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

        manager = SyncManager()
        manager.start()
        d = manager.dict()
        d['env'] = self.env.spec.id
        d['actor_type'] = type(self.actor)
        d['actor_config'] = self.actor.get_config()
        d['actor_weights'] = self.actor.get_weights()

        d['critic_type'] = type(self.critic)
        d['critic_config'] = self.critic.get_config()
        d['critic_weights'] = self.critic.get_weights()

        processes = [Process(target=worker, args=(d,)) for _ in range(num_threads)]

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
