Problems
========

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
:description: ...

HPO / XGBoost
-------------
:parameters:
    --folder   The folder containing the unpacked files of the `Steel Plates Faults <http://archive.ics.uci.edu/ml/datasets/Steel+Plates+Faults>`_ dataset. (required)
:dataset:        Dataset provided by Semeion, Research Center of Sciences of Communication, Via Sersale 117, 00128, Rome, Italy. www.semeion.it 
:dimensionality: :math:`135`
:description: ...
