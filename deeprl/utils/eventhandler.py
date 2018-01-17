class EventHandler(set):
    def __call__(self, *args, **kwargs):

        # If the function's instance has a .is_metric attribute = True then we know it's a metric
        metrics = set([e for e in self if getattr(e.__self__, 'is_metric', False)])

        # Call metrics first so their values are available to the remaining callbacks (i.e loggers)
        values = {event.__self__.name: event(**kwargs) for event in metrics}
        kwargs.update(values)

        # Call remaining events
        for event in self.difference(metrics):
            event(*args, **kwargs)

        return values

    def __iadd__(self, other):
        self.add(other)
        return self

    def __isub__(self, other):
        self.remove(other)
        return self




