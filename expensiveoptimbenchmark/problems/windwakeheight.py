from base import BaseProblem

import numpy as np
import floris.tools as wfct
import logging

class WindWakeHeightLayout(BaseProblem):

    def __init__(self, file, n_turbines=3, wind_seed=0, width=1000, length=1000, heights=[90, 95, 100, 105, 110], plant_kwv=None):
        self.file = file
        self.wind_seed = wind_seed
        self.n_turbines = n_turbines
        self.width = width
        self.length = length
        self.heights = heights
        self.wd, self.ws, self.freq = self._gen_random_wind(wind_seed)
        # self.loggerclass = logging.getLoggerClass()
        self.fi = wfct.floris_interface.FlorisInterface(self.file)
        # Set number of turbines
        rand_layout_x = np.random.uniform(0.0, self.width, size=n_turbines)
        rand_layout_y = np.random.uniform(0.0, self.length, size=n_turbines)
        

        self.fi.reinitialize_flow_field(layout_array=(rand_layout_x, rand_layout_y))

        # Default polygon (covers entire area)
        self.boundaries = [[0.0, 0.0], [width, 0.0], [width, length], [0.0, length]]
        # Scaling factor, set to 1 in order to avoid scaling.
        aep_initial = 1
        coe_initial = 1
        # Assume n_turbines x 5 MW...
        plant_kw = 5 * 1000 * n_turbines
        height_lb = np.min(heights)
        height_ub = np.max(heights)
        self.lo = wfct.optimization.scipy.layout_height.LayoutHeightOptimization(self.fi, self.boundaries, [height_lb, height_ub], self.wd, self.ws, self.freq, aep_initial, coe_initial, plant_kw)
        # Use the default minimum distance that floris themselves use.
        self.min_dist = self.lo.min_dist
        # logging.setLoggerClass(self.loggerclass)

    def _gen_random_wind(self, seed):
        rng = np.random.RandomState(seed)
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

        xc = np.copy(x)
        xc[-1] = self.heights[int(xc[-1])]

        obj = self.lo._COE_layout_height_opt(xc)
        return obj

    def lbs(self):
        return np.zeros(2 * self.n_turbines + 1, dtype=float)

    def ubs(self):
        # return (np.asarray([[self.width, self.length]]) * np.ones((self.n_turbines, 1), dtype=float)).ravel()
        return np.concatenate(np.ones(2 * self.n_turbines, dtype=float), np.asarray([len(self.heights) - 1]))

    def vartype(self):
        return np.array(['cont'] * self.dims() + ['int'])

    def dims(self):
        return self.n_turbines * 2 + 1

    def __str__(self):
        return f"WindWakeHeightLayout(file={self.file},n_turbines={self.n_turbines},wind_seed={self.wind_seed})"

    