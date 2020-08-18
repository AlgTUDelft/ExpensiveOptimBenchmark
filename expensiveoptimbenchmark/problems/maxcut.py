from .base import BaseProblem
import numpy as np
import networkx as nx
import os
from random import Random

class MaxCut(BaseProblem):

    def __init__(self, d, graph_seed = 42, noise_seed = None, noise_factor = 1):
        self.ub = 1
        self.lb = 0
        self.d = d
        self.noise_seed = noise_seed
        self.noise_rng = np.random.RandomState(seed=noise_seed)
        self.noise_factor = noise_factor

        # Always use same seed to "preserve" problem instances of a given dimension.
        # instantiate new Random class to keep the seed within this function
        r = Random(graph_seed)
        p = 0.5
        # Generate a (seeded) random graph with d nodes, edge density 0.9 and edge weight [0, 10]
        graph = nx.gnp_random_graph(d, p, seed=graph_seed)
        i = graph_seed + 1
        while not nx.is_connected(graph):
            print('Generated graph was disconnected. Re-creating graph')
            graph = nx.gnp_random_graph(d, p, seed=i)
            i += 1
        self.graph = {e: r.randint(0, 10) for e in graph.edges}

    def evaluate(self, x, minimize=True):
        assert len(x) == self.d

        objective = 0
        for v1, v2 in self.graph.keys():
            if x[v1] != x[v2]:
                objective += self.graph[(v1, v2)]

        if self.noise_seed is not True:
            # Add noise to objective. Mean=0, std=1.
            objective += self.noise_rng.normal(scale=self.noise_factor)

        if minimize is True:
            return -objective
        else:
            return objective

    def lbs(self):
        return self.lb*np.ones(self.d, dtype=int)

    def ubs(self):
        return self.ub*np.ones(self.d, dtype=int)

    def vartype(self):
        return np.array(['int'] * self.d)

    def dims(self):
        return self.d

    def __str__(self):
        return f"MaxCut(d={self.d})"