import numpy as np
import floris.tools as wfct
import logging

class WindWakeLayout:

    def __init__(self, file, n_turbines=3, wind_seed=0, width=1000, height=1000):
        self.file = file
        self.wind_seed = wind_seed
        self.n_turbines = n_turbines
        self.width = width
        self.height = height
        self.wd, self.ws, self.freq = self._gen_random_wind(wind_seed)
        # self.loggerclass = logging.getLoggerClass()
        self.fi = wfct.floris_interface.FlorisInterface(self.file)
        # logging.setLoggerClass(self.loggerclass)

    def _gen_random_wind(self, seed):
        rng = np.random.RandomState(seed)
        wd = np.arange(0.0, 360.0, 5.0)
        ws = 8.0 + rng.randn(len(wd)) * 0.5
        freq = np.abs(np.sort(rng.randn(len(wd))))
        freq = freq / freq.sum()
        return wd, ws, freq

    def evaluate(self, x):
        x_wrapped = x.reshape((self.n_turbines, 2))
        layout_x = x_wrapped[:, 0]
        layout_y = x_wrapped[:, 1]
        self.fi.reinitialize_flow_field(layout_array=(layout_x, layout_y))

        annual_energy_production = self.fi.get_farm_AEP(self.wd, self.ws, self.freq)
        # Reset logger.
        # logging.setLoggerClass(self.loggerclass)
        # Note: maximize production!
        return -1 * annual_energy_production

    def lbs(self):
        return np.zeros(2 * self.n_turbines, dtype=float)

    def ubs(self):
        return (np.asarray([[self.width, self.height]]) * np.ones((self.n_turbines, 1), dtype=float)).ravel()

    def vartype(self):
        return np.array(['cont'] * self.dims())

    def dims(self):
        return self.n_turbines * 2

    def __str__(self):
        return f"WindWakeLayout(file={self.file},n_turbines={self.n_turbines},wind_seed={self.wind_seed})"

    