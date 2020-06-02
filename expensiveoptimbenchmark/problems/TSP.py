import numpy as np

class TSP:

    def __init__(self, W, n_iter, noise_seed=None, noise_factor=1):
        self.d = W.shape[0] - 2
        self.W = W
        self.n_iter = n_iter
        self.noise_rng = np.random.RandomState(seed=noise_seed)
        self.noise_factor = noise_factor

    def _evaluate(self, x):
        robust_total_route_length = 0.0
        
        for iteration in range(self.n_iter):
            current = 0
            unvisited = list(range(1, self.d))

            for i in x:
                next_up = unvisited.pop(i)
                robust_total_route_length += self.W[current, next_up]
                current = next_up

            last = unvisited.pop()
            robust_total_route_length += self.W[current, last]
            robust_total_route_length += self.W[last, 0]

        robust_total_route_length += self.noise_rng.random() * self.n_iter * (self.d + 2) * self.noise_factor

        return robust_total_route_length

    def lbs(self):
        return np.zeros(self.d, dtype=int)

    def ubs(self):
        return np.array([self.d-x-1 for x in range(0, self.d)])

    def dims(self):
        return self.d


def load_explicit_W(path):
    with open(path) as f:
        return np.array([list(map(lambda x: float(x.strip()))) for line in f.readlines()])

import tsplib95

def load_tsplib_W(path):
    return np.array(tsplib95.load(path).edge_weight_format)
