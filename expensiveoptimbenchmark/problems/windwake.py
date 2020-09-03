from .base import BaseProblem

import numpy as np
import floris.tools as wfct
import logging

class WindWakeLayout(BaseProblem):

    def __init__(self, file, n_turbines=3, wind_seed=0, width=None, height=None, n_samples=5):
        self.file = file
        self.wind_seed = wind_seed
        self.n_turbines = n_turbines
        
        self.width = width if width is not None else 333.33 * n_turbines
        self.height = height if height is not None else 333.33 * n_turbines
        # How many times to sample the matrix. Default is to use a fixed sample.
        self.n_samples = n_samples
        self.wind_rng = np.random.RandomState(wind_seed)
        self.wd, self.ws, self.freq = self._gen_random_wind()
        # self.loggerclass = logging.getLoggerClass()
        self.fi = wfct.floris_interface.FlorisInterface(self.file)
        # Set number of turbines
        rand_layout_x = np.random.uniform(0.0, self.width, size=n_turbines)
        rand_layout_y = np.random.uniform(0.0, self.height, size=n_turbines)

        self.fi.reinitialize_flow_field(layout_array=(rand_layout_x, rand_layout_y))

        # Default polygon (covers entire area)
        self.boundaries = [[0.0, 0.0], [self.width, 0.0], [self.width, self.height], [0.0, self.height]]
        # Scaling factor, set to 1 in order to avoid scaling.
        self.aep_initial = 1
        self.lo = wfct.optimization.scipy.layout.LayoutOptimization(self.fi, self.boundaries, self.wd, self.ws, self.freq, self.aep_initial)
        # Use the default minimum distance that floris themselves use.
        self.min_dist = self.lo.min_dist
        # logging.setLoggerClass(self.loggerclass)

    def _gen_random_wind(self):
        rng = self.wind_rng
        wd = np.arange(0.0, 360.0, 5.0)
        ws = 8.0 + rng.randn(len(wd)) * 0.5
        freq = np.abs(np.sort(rng.randn(len(wd))))
        freq = freq / freq.sum()
        return wd, ws, freq

    def evaluate(self, x):
        c1 = self.lo._space_constraint(x, self.min_dist)
        c2 = self.lo._distance_from_boundaries(x, self.boundaries)

        # No power produced when constraints are violated.
        if c1 < 0 or c2 < 0:
            return 0.0

        if self.n_samples is None:
            obj = self.lo._AEP_layout_opt(x)
        else:
            obj = 0.0
            for _ in range(self.n_samples):
                # Resample wind speed
                self.ws = 8.0 + self.wind_rng.randn(len(self.wd)) * 0.5
                self.lo = wfct.optimization.scipy.layout.LayoutOptimization(self.fi, self.boundaries, self.wd, self.ws, self.freq, self.aep_initial)
                obj += self.lo._AEP_layout_opt(x)
            obj = obj / self.n_samples

        return obj

    def lbs(self):
        return np.zeros(2 * self.n_turbines, dtype=float)

    def ubs(self):
        # return (np.asarray([[self.width, self.height]]) * np.ones((self.n_turbines, 1), dtype=float)).ravel()
        return np.ones(2 * self.n_turbines, dtype=float)

    def vartype(self):
        return np.array(['cont'] * self.dims())

    def dims(self):
        return self.n_turbines * 2

    def __str__(self):
        return f"WindWakeLayout(file={self.file},n_turbines={self.n_turbines},width={self.width},height={self.height},wind_seed={self.wind_seed})"

    