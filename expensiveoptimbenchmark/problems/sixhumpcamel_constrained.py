from .base import BaseProblem
import numpy as np
#from scipy.optimize import rosen

class Sixhumpcamel_constrained(BaseProblem):

    def __init__(self, d_int, d_cont, lb=-2, ub=2, dolog=False, noise_seed=None, noise_factor=1e-6):
        self.ub = ub
        self.lb = lb
        self.d_int = d_int
        self.d_cont = d_cont
        d = d_int + d_cont
        self.d = d
        self.scaling = 1 #d*(100*((self.ub-self.lb**2)**2)+(self.ub-1)**2)
        self.dolog = dolog

        self.noise_seed = noise_seed
        self.noise_rng = np.random.RandomState(seed=noise_seed)
        self.noise_factor = noise_factor

    
        
    def evaluate(self, x):
        assert len(x) == self.d
        if self.d != 2:
            print('Warning: Six-hump camel function needs exactly 2 variables')
            
        # =============================================================================
        #  Six-hump Camel Function (f_min = - 1.0316 )
        #  https://www.sfu.ca/~ssurjano/camel6.html          
        # =============================================================================
        def mysixhumpcamp_constrained(X):
            X = np.asarray(X)
            X = np.reshape(X, (-1, 2))
            if len(X.shape) == 1:
                x1 = X[0]
                x2 = X[1]
            else:
                x1 = X[:, 0]
                x2 = X[:, 1]
            term1 = (4 - 2.1 * x1 ** 2 + (x1 ** 4) / 3) * x1 ** 2
            term2 = x1 * x2
            term3 = (-4 + 4 * x2 ** 2) * x2 ** 2
            fval = term1 + term2 + term3
            fval = fval[0]/10
            ## Return penalty if outside some specific polygon
            if x2>-x1+1.5 or x2<4/3*x1-2 or x2<-3*x1-2 or x2>0.5*x1+1.5:
                fval=1000
            return fval
        
        
        if self.dolog:
            return np.log(mysixhumpcamp_constrained(x) + 2)  + self.noise_rng.normal(scale=self.noise_factor)
        return mysixhumpcamp_constrained(x)/self.scaling + self.noise_rng.normal(scale=self.noise_factor)

    def lbs(self):
        return self.lb*np.ones(self.d, dtype=float)

    def ubs(self):
        return self.ub*np.ones(self.d, dtype=float)

    def vartype(self):
        return np.array(['int'] * self.d_int + ['cont'] * self.d_cont)

    def dims(self):
        return self.d

    def __str__(self):
        return f"Six-hump camel(d_int={self.d_int}, d_cont={self.d_cont}, log={self.dolog})"
