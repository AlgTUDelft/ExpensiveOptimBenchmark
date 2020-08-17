Problems
========

.. note::
    Evaluation runtimes are measured on a **Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz** CPU, using random search.

.. jupyter-execute::
    :hide-code:

    import os
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    is_doc = os.getcwd().endswith('doc')
    datadir = "./source/data/" if is_doc else "./data"

    def plot_time_hist(csv):
        esp_times = pd.read_csv(os.path.join(datadir, csv))['iter_eval_time']

        esp_times_n = len(esp_times)
        esp_times_min = np.min(esp_times)
        esp_times_mean = np.mean(esp_times)
        esp_times_std = np.std(esp_times)
        esp_times_max = np.max(esp_times)
        
        fig = plt.figure()
        ax = fig.add_subplot()
        
        # Label axes
        ax.set_ylabel("Count")
        ax.set_xlabel("Time (s)")

        # Add histogram
        ax.hist(esp_times)

        # Overlay gaussian
        x = np.linspace(esp_times_min, esp_times_max, 200)
        y = (esp_times_n / (esp_times_std * np.sqrt(2))) * np.exp(-1/2*((x - esp_times_mean) / esp_times_std)**2)
        ax.plot(x, y)

        ax.text(0.01, 0.99, f"$\\mu: {esp_times_mean:.2f}$\n$\sigma: {esp_times_std:.2f}$", horizontalalignment='left',
        verticalalignment='top', transform=ax.transAxes)

        fig.show()

Windmill Wake Simulator
-----------------------
:bibtex:     ``floris2020``
:repository:  `GitHub <https://github.com/NREL/floris>`_
:parameters:
    -n   The number of windmills to be placed. (default: 3)
    -w   The width of the area in which the windmills are to be placed. (default: 1000)
    -h   The height of the area in which the windmills are to be placed. (default: 1000)
    --wind-seed  The random seed used for generating the distribution and strength of the wind. (default: 0)
:dimensionality: :math:`2n`
:description: ...

Electrostatic Precipitator*
---------------------------
:publications: :cite:`daniels2018suite`
:bibtex:      ``daniels2018suite``
:repository:   `BitBucket <https://bitbucket.org/arahat/cfd-test-problem-suite/>`_
:parameters:    None
:dimensionality: :math:`49`
:runtime:
    .. jupyter-execute::
        :hide-code:

        plot_time_hist("esp_rs.csv.xz")

:description: ...

HPO / XGBoost
-------------
:parameters:
    --folder   The folder containing the unpacked files of the `Steel Plates Faults <http://archive.ics.uci.edu/ml/datasets/Steel+Plates+Faults>`_ dataset. (required)
:dataset:        Dataset provided by Semeion, Research Center of Sciences of Communication, Via Sersale 117, 00128, Rome, Italy. www.semeion.it 
:dimensionality: :math:`135`
:runtime:
    .. jupyter-execute::
        :hide-code:

        plot_time_hist("hpo_rs.csv.xz")

:parameters:
    --time-limit   The time limit for a single evaluation of the objective function in seconds. (default: 8)
        **TODO:** Setting this parameter still needs to be implemented.

        .. important::
            The default time limit is based on a **Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz**, adjust accordingly to hardware used.
        
:description: ...
