class EventHandler(set):

    def __call__(self, *args, **kwargs):
        for event in self:
            event(*args, **kwargs)

    def __iadd__(self, other):
        self.add(other)
        return self

    def __isub__(self, other):
        self.remove(other)
        return self




