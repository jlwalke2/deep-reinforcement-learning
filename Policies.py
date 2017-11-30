import numpy as np

class BoltzmannPolicy():
    def __init__(self, min_temp=0.0, max_temp=100):
        self.min_temp = min_temp
        self.max_temp = max_temp

    def __call__(self, qvalues, *args, **kwargs):
        probs = np.round(np.exp(qvalues.astype('float64')), 5)  # Convert to float64 to avoid overflow from exp
        probs /= np.sum(probs)              # Normalize to sum to 1
        probs[0] -= (np.sum(probs) - 1.0)   # Total probability can be close but != 1.0.  +/- any difference arbitrarily to the first action
        np.clip(probs, 0., 1., out=probs)   # Ensure no probability outside 0..1
        probs = probs.ravel()

        return np.random.choice(range(qvalues.size), p=probs)