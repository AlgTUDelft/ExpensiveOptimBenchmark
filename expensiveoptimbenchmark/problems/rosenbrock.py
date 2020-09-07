from .base import BaseProblem
import numpy as np
from scipy.optimize import rosen

class Rosenbrock(BaseProblem):

    def __init__(self, d_int, d_cont, lb=-5, ub=10, dolog=False, noise_seed=None, noise_factor=1e-6):
        self.ub = ub
        self.lb = lb
        self.d_int = d_int
        self.d_cont = d_cont
        d = d_int + d_cont
        self.d = d
        self.scaling = d*(100*((self.ub-self.lb**2)**2)+(self.ub-1)**2)
        self.dolog = dolog

        self.noise_seed = noise_seed
        self.noise_rng = np.random.RandomState(seed=noise_seed)
        self.noise_factor = noise_factor

    def evaluate(self, x):
        assert len(x) == self.d
        if self.dolog:
            return np.log(rosen(x) + 1)  + self.noise_rng.normal(scale=self.noise_factor)
        return rosen(x)/self.scaling + self.noise_rng.normal(scale=self.noise_factor)

    def lbs(self):
        return self.lb*np.ones(self.d, dtype=float)

    def ubs(self):
        return self.ub*np.ones(self.d, dtype=float)

    def vartype(self):
        return np.array(['int'] * self.d_int + ['cont'] * self.d_cont)

    def dims(self):
        return self.d

    def __str__(self):
        return f"Rosenbrock(d_int={self.d_int}, d_cont={self.d_cont}, log={self.dolog})"
