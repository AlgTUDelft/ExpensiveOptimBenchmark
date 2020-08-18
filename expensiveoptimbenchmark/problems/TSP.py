from .base import BaseProblem
import os.path
import numpy as np
import tsplib95
import networkx

class TSP(BaseProblem):

    def __init__(self, name, W, n_iter, noise_seed=None, noise_factor=1):
        self.name = name
        self.d = W.shape[0] - 2
        self.W = W
        self.n_iter = n_iter
        self.noise_seed = noise_seed
        self.noise_rng = np.random.RandomState(seed=noise_seed)
        self.noise_factor = noise_factor

    def evaluate(self, x):
        robust_total_route_length = 0.0
        
        for iteration in range(self.n_iter):
            current = 0
            unvisited = list(range(1, self.d+2))
            total_route_length = 0.0

            for di, i in enumerate(x):
                next_up = unvisited.pop(int(round(i)))
                total_route_length += self.W[current, next_up]
                total_route_length += self.noise_rng.random() * self.noise_factor
                current = next_up

            last = unvisited.pop()
            total_route_length += self.W[current, last]
            total_route_length += self.noise_rng.random() * self.noise_factor
            total_route_length += self.W[last, 0]
            total_route_length += self.noise_rng.random() * self.noise_factor

            robust_total_route_length = max(total_route_length, robust_total_route_length)
        
        return robust_total_route_length

    def lbs(self):
        return np.zeros(self.d, dtype=int)

    def ubs(self):
        return np.array([self.d-x for x in range(0, self.d)])

    def vartype(self):
        return np.array(['int'] * self.d)

    def dims(self):
        return self.d

    def __str__(self):
        return f"TSP(name={self.name},iterations={self.n_iter},noise_seed={self.noise_seed})"


def load_explicit_tsp(path, iters=100, noise_seed=0):
    with open(path) as f:
        W = np.array([list(map(lambda x: float(x.strip()))) for line in f.readlines()])
        return TSP(os.path.basename(path), W, iters, noise_seed=noise_seed)

def load_tsplib(path, iters=100, noise_seed=0):
    instance = tsplib95.load(path)
    W = networkx.to_numpy_matrix(instance.get_graph())
    return TSP(instance.name, W, iters, noise_seed=noise_seed)
