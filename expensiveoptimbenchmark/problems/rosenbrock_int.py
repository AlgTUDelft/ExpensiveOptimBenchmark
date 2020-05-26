import numpy as np
from scipy.optimize import rosen

class RosenbrockInt:

    def __init__(self, d):
        self.ub = 10
        self.lb = -5
        self.d = d
        self.scaling = d*(100*((self.ub-self.lb**2)**2)+(self.ub-1)**2)

    def _evaluate(self, x):
        return rosen(x)/self.scaling

    def f_kw(self, **kargs):
        vc = np.array([v for k, v in kargs.items()])
        return self._evaluate(vc)
        
    def f_arr(self, x):
        return self._evaluate(x)

    def lbs(self):
        return self.lb*np.ones(self.d, dtype=int)

    def ubs(self):
        return self.ub*np.ones(self.d, dtype=int)

    def n(self):
        return self.d

    def vars(self):
        return {
            'i{x}'.format(x=x): ('int', [self.lb, self.ub]) for x in range(0, self.d)
        }