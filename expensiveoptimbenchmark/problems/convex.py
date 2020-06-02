import numpy as np

class Convex:

    def __init__(self, d, instance_seed, noise_seed=None, noise_factor=1.0):
        self.d = d

        # Seed the rng that will produce the instance (x_star, A) 
        rng = np.random.RandomState(seed=instance_seed)

        # Generate optimal solution x_star
        self.x_star = rng.choice((0, 1), d)

        # Compute the positive semidefinite matrix A.
        U = rng.uniform(0.0, 1.0, (d, d))
        self.A = (U + U.T) / d + np.identity(d)

        # Prepare the noise. Set the noise_seed to anything other than None
        # to obtain a function that is deterministic (though varying call-by-call)
        self.noise_factor = noise_factor
        self.noise_rng = np.random.RandomState(seed=noise_seed)

    def _evaluate(self, x):
        diff = x - self.x_star
        return np.matmul(np.matmul(diff.T, self.A), diff) + self.noise_rng.uniform() * self.noise_factor

    def f_kw(self, **kargs):
        vc = np.array([v for k, v in kargs.items()])
        return self._evaluate(vc)
        
    def f_arr(self, x):
        return self._evaluate(x)

    def lbs(self):
        return np.zeros(self.d, dtype=int)

    def ubs(self):
        return np.ones(self.d, dtype=int)

    def n(self):
        return self.d

    def vars(self):
        return {
            'i{x}'.format(x=x): ('int', [0, 1]) for x in range(0, self.d)
        }