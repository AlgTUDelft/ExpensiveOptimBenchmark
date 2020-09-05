Problems
========
All evaluation functions provided by the benchmark suite are required to be minimized. If the objective value of the original underlying problem is to be maximized, the evaluation function provided has its value multiplied by :math:`-1`.

.. note::
    Evaluation runtimes are measured on a **Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz** CPU, using random search.

.. jupyter-execute::
    :hide-code:

    import os
    import pandas as pd
    import numpy as np
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt
    from scipy.stats import norm, gamma, chi, chi2, beta, expon

    is_doc = os.getcwd().endswith('doc')
    datadir = "./source/data/" if is_doc else "./data"

    def plot_hist(csv, field, dist=None, filter=None, scale=1):
        data = pd.read_csv(os.path.join(datadir, csv))[field]

        data_n = len(data)
        data_min = np.min(data)
        data_mean = np.mean(data)
        data_std = np.std(data)
        data_max = np.max(data)
        bins = 20
        do_plot_density = dist is not None
        
        fig = plt.figure()
        ax = fig.add_subplot()
        
        # Label axes
        if do_plot_density:
            ax.set_ylabel("Density")
        else:
            ax.set_ylabel("Count")
        if 'time' in field:
            ax.set_xlabel("Time (s)")

        # Add histogram
        ax.hist(data, bins=bins, density=do_plot_density)

        # Overlay distribution
        if dist is not None:
            x = np.linspace(data_min, data_max, 200)
            if filter is None:
                data_fil = data
            else:
                data_fil = data[filter(data)]

            dist_f = dist(*dist.fit(data_fil * scale))

            data_f_n = len(data_fil)
            y = dist_f.pdf(x * scale)
            ax.plot(x, y)

        txt = ax.text(0.01, 0.99, f"$\\mu: {data_mean:.2f}$\n$\sigma: {data_std:.2f}$\n$c: {data_n}$", horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)
        txt.set_path_effects([pe.Stroke(linewidth=3, foreground='white'), pe.Normal()])

Windmill Wake Simulator
-----------------------
:bibtex:     ``floris2020``
:repository:  `GitHub <https://github.com/NREL/floris>`__
:parameters:
    --file   The path to the windpark/windmill specification. We recommend using `example_input.json <https://raw.githubusercontent.com/NREL/floris/master/examples/example_input.json>`__ from the FLORIS repository. (required)
    -n   The number of windmills to be placed. (default: 3)
    -w   The width of the area in which the windmills are to be placed. (default: 333.33 * ``-n``)
    -h   The height of the area in which the windmills are to be placed. (default: 333.33 * ``-n``)
    --wind-seed  The random seed used for generating the distribution and strength of the wind. (default: 0)
    --n-samples  The number of random wind strength samples to evaluate. More is less noisy but takes more time. Passing the string ``None`` will use a fixed set of wind strengths (previous behaviour, fast, no noise) (default: 5)
:dimensionality: :math:`2n`, all continuous (``cont``)
:constraints: Windmills are not allowed to be located within a factor of two of each others' radius, this constraint has been incorporated into the objective function. Violations will result in an objective value of :math:`0.0`.
:description: The layout of the windmills in a wind farm has noticeable impact on the amount of energy it produces. This benchmark problem employs the `FLORIS <https://github.com/NREL/floris>`__ wake simulator to analyse how much power production is lost by having windmills be located in each others wake. The objective is to maximize power production.
:runtime:
    **At ``-n`` = 3:**

    .. jupyter-execute::
        :hide-code:

        plot_hist("windwake_rs.csv.xz", 'iter_eval_time', dist=norm)

    **At ``-n`` = 5:**

    .. jupyter-execute::
        :hide-code:

        plot_hist("windwake_rs_5.csv.xz", 'iter_eval_time', dist=norm)

:fitness:
    **At ``-n`` = 3:**

    .. jupyter-execute::
        :hide-code:

        plot_hist("windwake_rs.csv.xz", 'iter_fitness')

    **At ``-n`` = 5:**

    .. jupyter-execute::
        :hide-code:

        plot_hist("windwake_rs_5.csv.xz", 'iter_fitness')


Electrostatic Precipitator*
---------------------------
:publications: (:cite:`daniels2018suite`)
:bibtex:      (``daniels2018suite``)
:repository:   `BitBucket <https://bitbucket.org/arahat/cfd-test-problem-suite/>`__
:parameters:    None
:dimensionality: :math:`49` - all categorical (``cat``)

:runtime:
    .. jupyter-execute::
        :hide-code:

        plot_hist("esp_rs.csv.xz", 'iter_eval_time', dist=norm)

:fitness:
    .. jupyter-execute::
        :hide-code:

        plot_hist("esp_rs.csv.xz", 'iter_fitness')

:description: An Electrostatic Precipitator is a large gas filtering installation, whose efficiency and efficiacy is dependent on how well the intake gas is distributed. This installation has slots -- named baffles -- which can be of various types, each having a different impact on the distribution. This benchmark problem employs the OpenFOAM Computational Fluid Dynamics simulator, implemented as part of the `CFD Test Problem Suite <https://bitbucket.org/arahat/cfd-test-problem-suite/>`__ by Daniels et al. . The goal is to find a configuration that has the best resulting distribution.

PitzDaily
---------
:publications: :cite:`daniels2018suite`
:bibtex:      ``daniels2018suite``
:repository:   `BitBucket <https://bitbucket.org/arahat/cfd-test-problem-suite/>`__
:parameters:    None
:dimensionality: :math:`10` - all continuous (``cont``)

:runtime:
    .. jupyter-execute::
        :hide-code:

        plot_hist("pitzdaily_rs.csv.xz", 'iter_eval_time', dist=norm)

:fitness:
    .. jupyter-execute::
        :hide-code:

        plot_hist("pitzdaily_rs.csv.xz", 'iter_fitness')

:constraints: Points must lie in a polygon, constraint violations will result in an objective value of :math:`1.0`.

:description: 

HPO / XGBoost
-------------
:parameters:
    --folder   The folder containing the unpacked files of the `Steel Plates Faults <http://archive.ics.uci.edu/ml/datasets/Steel+Plates+Faults>`__ dataset. (required)
    --time-limit   The time limit for a single evaluation of the objective function in seconds.
        A that requires more time than what time time limit allows will return an objective value of 0 (default: 8)
        **TODO:** Setting this parameter still needs to be implemented.

        .. important::
            The default time limit is based on a **Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz**, adjust accordingly to hardware used.
        
:dataset:        Dataset provided by Semeion, Research Center of Sciences of Communication, Via Sersale 117, 00128, Rome, Italy. www.semeion.it 
:dimensionality: :math:`135` - :math:`117` categorical (``cat``), :math:`7` integer (``int``), :math:`11` continuous (``cont``), contains conditionals
:runtime:
    .. jupyter-execute::
        :hide-code:

        # plot_hist("hpo_rs.csv.xz", 'iter_eval_time', dist=gamma, filter=lambda x: x < 8.0)
        plot_hist("hpo_rs.csv.xz", 'iter_eval_time', dist=norm)
    
:fitness:
    .. jupyter-execute::
        :hide-code:

        plot_hist("hpo_rs.csv.xz", 'iter_fitness')

:constraints: Time it limited to 8s (on our machine), violations result in an objective value of :math:`0.0`.

:description: Machine Learning approaches often have a large amount of hyperparameters of varying types. This benchmark makes use of scikit-learn to build an XGBoost classifier with per-feature preprocessing. Evaluation of a solution is performed by k-fold cross validation, with the goal to maximize accuracy.

Rosenbrock
----------
:parameters:
    --n-int   The number of dimensions that are required to be integer (expressed as :math:`i` in the dimensionality below)
    --n-cont   The number of dimensions that are required to be continuous (expressed as :math:`c` in the dimensionality below)
    --logscale   Whether to take the log of the rosenbrock function instead of scaling.
:dimensionality: :math:`i + c`, :math:`i` integer (``int``), :math:`c` continuous (``cont``)
:description: The rosenbrock function with a configurable amount of integer and continuous variables. Non-expensive problem included to test whether approaches work.