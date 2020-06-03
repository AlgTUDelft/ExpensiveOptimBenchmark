import numpy as np

class Convex:

    def __init__(self, d, instance_seed, noise_seed=None, noise_factor=1.0):
        self.d = d

        # Seed the rng that will produce the instance (x_star, A) 
        self.instance_seed = instance_seed
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

    def evaluate(self, x):
        diff = x - self.x_star
        return np.matmul(np.matmul(diff.T, self.A), diff) + self.noise_rng.uniform() * self.noise_factor

    def lbs(self):
        return np.zeros(self.d, dtype=int)

    def ubs(self):
        return np.ones(self.d, dtype=int)

    def vartype(self):
        return np.array(['int'] * self.d)

    def dims(self):
        return self.d

    def __str__(self):
        return f"Convex(d={self.d}, seed={self.instance_seed})"
