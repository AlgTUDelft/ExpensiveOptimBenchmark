import numpy as np
from scipy.optimize import rosen

class RosenbrockBinarized:

    def __init__(self, d):
        self.unbinarized_ub = 10
        self.unbinarized_lb = -5
        self.unbinarized_d = d

        self.ub = 1  # Unbinarized would be 10
        self.lb = 0  # Unbinarized would be -5
        self.d = 4*d # Length of interval is 2^4=16
        self.scaling = d*(100*((self.unbinarized_ub-self.unbinarized_lb**2)**2)+(self.unbinarized_ub-1)**2)

    def evaluate(self, x):
        assert len(x) == self.d
        
        # Compute non-binary representation
        unbinarized_x = []
        for i in range(self.unbinarized_d):
            x_i = 0
            for j in x[4*i:4*i+4]:
                x_i = (x_i << 1) | j # bitshift to convert binary to integer
            normalized_x = x_i+self.unbinarized_lb
            unbinarized_x.append(normalized_x)
            
        return rosen(unbinarized_x)/self.scaling
        
    def lbs(self):
        return self.lb*np.ones(self.d, dtype=int)

    def ubs(self):
        return self.ub*np.ones(self.d, dtype=int)

    def vartype(self):
        return np.array(['int'] * self.d)

    def dims(self):
        return self.d

    def __str__(self):
        return f"RosenbrockBinary(d={self.unbinarized_d})"
