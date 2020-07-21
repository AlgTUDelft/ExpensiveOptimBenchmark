# Code received from MiVaBO author Erik Daxberger, 23-03-2020
# Modified to work with the general benchmark layout:
# - Added lower bounds, upper bounds, variable type, dimensionality and descriptor functions. 

from itertools import combinations
import numpy as np


class Linear():
    """ Function that is linear in arbitrary features of discrete and continuous variables """

    def __init__(
        self,
        n_vars=16,      # total number of variables
        n_vars_d=8,     # number of discrete variables
        alpha=1.0,      # prior precision
        beta=1.0,       # observation noise precision
        sigma=1.0,      # kernel lengthscale / bandwidth
        n_feats_c=16,   # number of continuous features
        noisy=False,    # should we add observation noise?
        laplace=True,   # should we sample the weights from a Laplace distribution?
        seed=None,      # Seed for the rng.
    ):
        # set variables
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.n_vars_d = n_vars_d
        self.n_vars_c = n_vars - self.n_vars_d
        self.alpha = alpha # Store for descriptor
        self.beta = beta
        self.sigma = sigma # Store for descriptor
        self.n_feats_c = n_feats_c
        self.noisy = noisy
        self.vars_d_sq = list(combinations(range(self.n_vars_d), r=2))
        self.n_feats_d = self.n_vars_d + len(self.vars_d_sq)
        self.sample_feats_c(sigma)
        n_feats_m = self.n_feats_d * self.n_feats_c
        self.n_feats_total = 1 + self.n_feats_d + self.n_feats_c + n_feats_m
        self.laplace = laplace # Store for descriptor

        # sample the coefficients
        self.sample_coeffs(alpha, laplace)

    def sample_coeffs(self, alpha, laplace):
        """ Sample the coefficients from either a Laplace or a Gaussian distribution """

        if laplace:
            self.w = self.rng.laplace(0.0, 1.0 / alpha, self.n_feats_total)
            # ensure sparsity by setting small weights to zero
            self.w[self.w < 1.0 / alpha] = 0.0
        else:
            self.w = self.rng.normal(0.0, 1.0 / alpha, self.n_feats_total)

        # extract the coefficients
        self.w0 = self.w[0]
        self.w_d = self.w[1 : 1 + self.n_feats_d]
        self.w_c = self.w[1 + self.n_feats_d : 1 + self.n_feats_d + self.n_feats_c]
        self.w_m = self.w[1 + self.n_feats_d + self.n_feats_c : self.n_feats_total]

    def sample_feats_c(self, sigma):
        """ sample the continuous feature parameters, i.e.,
            random Fourier feature / random kitchen sink parameters U and b """

        self.rks_U = self.rng.normal(size=(self.n_feats_c, self.n_vars_c)) * (1.0 / sigma)
        self.rks_b = 2.0 * np.pi * np.random.rand(self.n_feats_c)
        self.rks_c = np.sqrt(2.0 / self.n_feats_c)

    def phi_c(self, x_c):
        """ basis functions / features for the continuous variables:
        random Fourier features / random kitchen sinks """

        return self.rks_c * np.cos(np.matmul(self.rks_U, x_c) + self.rks_b)

    def phi_d(self, x_d):
        """ basis functions / features for the discrete variables:
        (discrete) Fourier basis functions (-> 2nd order multi-linear polynomial) """

        phi = [x_d[i] for i in range(self.n_vars_d)]
        phi += [x_d[i] * x_d[j] for (i, j) in self.vars_d_sq]
        return np.array(phi)

    def phi_m(self, x_d, x_c):
        """ mixed basis functions / features:
            pairwise combinations of discrete and continuous features """

        return np.ndarray.flatten(np.outer(self.phi_d(x_d), self.phi_c(x_c)))

    def objective_function(self, x):
        """ objective function
            f(x) = w0 + f_d(x_d) + f_c(x_c) + f_m(x_d, x_c) """

        w0 = self.w0
        f_d = self.f_d(x[: self.n_vars_d])
        f_c = self.f_c(x[self.n_vars_d :])
        f_m = self.f_m(x[: self.n_vars_d], x[self.n_vars_d :])
        f = w0 + f_d + f_c + f_m

        return f if not self.noisy else self.rng.normal(f, 1 / self.beta)

    def f_d(self, x_d):
        """ linear model of discrete features """
        return np.dot(self.phi_d(x_d), self.w_d)

    def f_c(self, x_c):
        """ linear model of continuous features """
        return np.dot(self.phi_c(x_c), self.w_c)

    def f_m(self, x_d, x_c):
        """ linear model mixed features """
        return np.dot(self.phi_m(x_d, x_c), self.w_m)

    # For compatibility
    def evaluate(self, x):
        return self.objective_function(x)

    def lbs(self):
        return np.zeros(self.dims())

    def ubs(self):
        return np.ones(self.dims()) * 3

    def vartype(self):
        return np.array(['int'] * self.n_vars_d + ['cont'] * self.n_vars_c)

    def dims(self):
        return self.n_vars_d + self.n_vars_c

    def __str__(self):
        return f"LinearMIVABO(seed={self.seed},n_vars_d={self.n_vars_d},n_vars_c={self.n_vars_c},alpha={self.alpha},beta={self.beta},sigma={self.sigma},n_feats_d={self.n_feats_d},n_feats_c={self.n_feats_c},noisy={self.noisy},laplace={self.laplace})"

