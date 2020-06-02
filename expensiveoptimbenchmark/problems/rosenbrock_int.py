import numpy as np
from scipy.optimize import rosen

class RosenbrockInt:

    def __init__(self, d):
        self.ub = 10
        self.lb = -5
        self.d = d
        self.scaling = d*(100*((self.ub-self.lb**2)**2)+(self.ub-1)**2)

    def evaluate(self, x):
        return rosen(x)/self.scaling

    def lbs(self):
        return self.lb*np.ones(self.d, dtype=int)

    def ubs(self):
        return self.ub*np.ones(self.d, dtype=int)

    def vartype(self):
        return np.array(['int'] * self.d)

    def dims(self):
        return self.d
