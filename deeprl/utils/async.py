# from multiprocessing.managers import SyncManager
import rpyc
import pickle

class SharedModelService(rpyc.Service):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.config = dict()

    def on_connect(self, conn):
        conn._config.update(dict(allow_pickle=True,
                         allow_all_attrs=True,
                         allow_public_attrs=True,
                         allow_setattr=True,
                         allow_getattr=True))

    def on_disconnect(self, conn):
        pass

    def create_model(self, name, model_type, model_config, optimizer_type, optimizer_config, weights):
        import pickle

        weights = pickle.loads(pickle.dumps(weights))

        model_config = pickle.loads(pickle.dumps(model_config))
        optimizer_config = pickle.loads(pickle.dumps(optimizer_config))

        optimizer_type = pickle.loads(pickle.dumps(optimizer_type))
        model = SharedModel(model_type, model_config, optimizer_type, optimizer_config, weights)
        self.config[name] = model



class SharedModel(object):
    def __init__(self, model_type, model_config, optimizer_type, optimizer_config, weights):
        self.model_type = model_type
        self.model_config = model_config
        self.weights = weights
        self.optimizer_type = optimizer_type
        self.optimizer_config = optimizer_config

    def get_weights(self):
        return self.weights

    def add_delta(self, deltas):
        deltas = pickle.loads(pickle.dumps(deltas))
        self.weights = [w + d for w, d in zip(self.weights, deltas)]
        return self.weights

    def get_model(self):
       return self.model_type, self.model_config

    def get_optimizer(self):
        return self.optimizer_type, self.optimizer_config


# class ModelManager(SyncManager):
#     pass
#
# ModelManager.register('Model', SharedModel)
# ModelManager.register('Monitor', History)