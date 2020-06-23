import numpy as np
from scipy.optimize import rosen

class Rosenbrock:

    def __init__(self, d):
        self.ub = 10.0
        self.lb = -5.0
        self.d = d
        self.scaling = d*(100*((self.ub-self.lb**2)**2)+(self.ub-1)**2)

    def evaluate(self, x):
        assert len(x) == self.d
        return rosen(x)/self.scaling

    def lbs(self):
        return self.lb*np.ones(self.d)

    def ubs(self):
        return self.ub*np.ones(self.d)

    def vartype(self):
        return np.array(['cont'] * self.d)

    def dims(self):
        return self.d

    def __str__(self):
        return f"Rosenbrock(d={self.d})"
