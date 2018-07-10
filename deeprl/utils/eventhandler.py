import logging

logger = logging.getLogger(__name__)

class EventHandler():
    def __init__(self):
        self.metrics = set()
        self.callbacks = set()

    def __call__(self, **kwargs):
        values = {}

        # Call all metrics first and include the results when calling the remaining callbacks.
        for metric in self.metrics:
            # Call the metric and add the result to the arguments passed to callbacks
            # If the method signature is not correct (ie missing **kwargs) a TypeError is thrown.
            # Raise an error to explicitly indicate which event caused the failure.
            try:
                values.update(metric(**kwargs))
            except TypeError:
                raise TypeError(f'An unhandled exception was raised while executing {metric}.')

        kwargs.update(values)

        # Call remaining functions
        # If the method signature is not correct (ie missing **kwargs) a TypeError is thrown.
        # Raise an error to explicitly indicate which event caused the failure.
        for event in self.callbacks:
            try:
                event(**kwargs)
            except TypeError:
                raise TypeError(f'An unhandled exception was raised while calling {event}.')

        return values

    def __contains__(self, item):
        return item in self.metrics or item in self.callbacks

    def __iter__(self):
        for obj in self.metrics.union(self.callbacks):
            yield obj

    def __len__(self):
        return len(self.metrics) + len(self.callbacks)

    def __iadd__(self, other):
        if getattr(other, 'is_metric', False):
            self.metrics.add(other)
        else:
            self.callbacks.add(other)

        return self

    def __isub__(self, other):
        if other in self.callbacks:
            self.callbacks.remove(other)

        if other in self.metrics:
            self.metrics.remove(other)

        return self




