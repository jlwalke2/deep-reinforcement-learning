from multiprocessing.managers import SyncManager

class SharedModel():
    def __init__(self, model_type, model_config, optimizer_type, optimizer_config, weights):
        self.model_type = model_type
        self.model_config = model_config
        self.weights = weights
        self.optimizer_type = optimizer_type
        self.optimizer_config = optimizer_config

    def get_weights(self):
        return self.weights

    def add_delta(self, deltas):
        self.weights = [w + d for w, d in zip(self.weights, deltas)]
        return self.weights

    def get_model(self):
        return self.model_type, self.model_config

    def get_optimizer(self):
        return self.optimizer_type, self.optimizer_config


class ModelManager(SyncManager):
    pass

ModelManager.register('Model', SharedModel)