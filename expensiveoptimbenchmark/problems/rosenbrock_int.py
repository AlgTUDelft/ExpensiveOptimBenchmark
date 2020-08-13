import numpy as np
from scipy.optimize import rosen

class RosenbrockInt:

    def __init__(self, d, dolog=False, noise_seed=None, noise_factor=1e-6):
        self.ub = 10
        self.lb = -5
        self.d = d
        self.scaling = d*(100*((self.ub-self.lb**2)**2)+(self.ub-1)**2)
        self.dolog = dolog

        self.noise_seed = noise_seed
        self.noise_rng = np.random.RandomState(seed=noise_seed)
        self.noise_factor = noise_factor

    def evaluate(self, x):
        assert len(x) == self.d
        if self.dolog:
            return np.log(rosen(x) + 1)  + self.noise_rng.random() * self.noise_factor
        return rosen(x)/self.scaling + self.noise_rng.random() * self.noise_factor

    def lbs(self):
        return self.lb*np.ones(self.d, dtype=int)

    def ubs(self):
        return self.ub*np.ones(self.d, dtype=int)

    def vartype(self):
        return np.array(['int'] * self.d)

    def dims(self):
        return self.d

    def __str__(self):
        return f"RosenbrockInt(d={self.d}, log={self.dolog})"
