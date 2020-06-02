import numpy as np

class TSP:

    def __init__(self, W, n_iter, noise_factor):
        self.d = W.shape[0] - 2
        self.W = W
        self.n_iter = n_iter
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

        robust_total_route_length += np.random.random() * self.n_iter * (self.d + 2) * self.noise_factor

        return robust_total_route_length

    def f_kw(self, **kargs):
        vc = np.array([v for k, v in kargs.items()])
        return self._evaluate(vc)
        
    def f_arr(self, x):
        return self._evaluate(x)

    def lbs(self):
        return np.zeros(self.d, dtype=int)

    def ubs(self):
        return np.array([self.d-x-1 for x in range(0, self.d)])

    def n(self):
        return self.d

    def vars(self):
        return {
            'i{x}'.format(x=x): ('int', [0, self.d-x-1]) for x in range(0, self.d)
        }