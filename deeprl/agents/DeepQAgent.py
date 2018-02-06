from .abstract import AbstractAgent


class DeepQAgent(AbstractAgent):
    def __init__(self, model, *args, **kwargs):
        super(DeepQAgent, self).__init__(*args, **kwargs)
        self.model = model
        self.preprocess_steps = []


    def train(self, steps_before_training: int =None, **kwargs):
        # Unless otherwise specified, assume training doesn't start until a full sample of steps is observed
        self._steps_before_training = steps_before_training or self.memory.sample_size

        super(DeepQAgent, self).train(**kwargs)
