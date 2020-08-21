Approaches
==========

HyperOpt
--------
:publications: :cite:`bergstra2013making`.
:bibtex:      ``bergstra2013making``
:repository:   `GitHub <https://github.com/hyperopt/hyperopt>`_, `PyPI <https://pypi.org/project/hyperopt/>`_
:supports:    ``cont``, ``int``, ``cat``
:parameters:
    --rand-evals   The number of random evaluations to perform before utilizing the surrogate model. (default: 3)
    --int-conversion-mode   How to convert integer variables. Choose from ``quniform`` or ``randint``. (default: ``quniform``)

Random Search (via HyperOpt)
----------------------------
:publications: :cite:`bergstra2013making`.
:bibtex:      ``bergstra2013making``
:supports:    ``cont``, ``int``, ``cat``
:repository:   `GitHub <https://github.com/hyperopt/hyperopt>`_, `PyPI <https://pypi.org/project/hyperopt/>`_

SMAC3
-----
:publications: :cite:`hutter2010sequential-extended`, :cite:`hutter2011sequential`
:bibtex:      ``hutter2010sequential-extended``, ``hutter2011sequential``
:repository:   `GitHub <https://github.com/automl/SMAC3>`_, `PyPI <https://pypi.org/project/smac/>`_
:supports:    ``cont``, ``int``, ``cat``
:parameters:
    --rand-evals   The number of random evaluations to perform before utilizing the surrogate model. (default: 3)
    --deterministic   Whether to run SMAC in deterministic mode. (default: false)

        .. warning:: Non-deterministic mode spends a significant portion of its evaluation budget re-evaluating previous solutions. Enabling deterministic mode disables this, and may therefore provide better performance.
        

IDONE
-----
:publications: :cite:`bliek2019black` 
:bibtex:       ``bliek2019black``
:repository:   `BitBucket <https://bitbucket.org/lbliek2/idone>`_
:supports:   ``int``, ``cat`` (as interpreted as integer / binarized)
:parameters:
    --rand-evals   The number of random evaluations to perform before utilizing the surrogate model. (default: 5)
    --model   The kind of model to use. Choose between ``basic`` and ``advanced``. (default: ``advanced``)
    --binarize-categorical   Whether to binarize categorical variables. Will turn a categorical variable with :math:`k` possible values, into :math:`\log_2(k)` binary (0 or 1) categorical variables. (default: false)
    --binarize-int   Whether to binarize integer variables. Similar to ``--binarize-categorical``, will turn a integer variable with :math:`k` possible values, into :math:`\log_2(k)` binary (0 or 1) integer variables. (default: false)
    --sampling   What kind of random sampling to perform to motivate exploration. Can be ``none``, ``thompson`` or ``uniform``. (default: ``none``)
    --scaling   Whether to perform scaling based on the first sample. (default: false)
    --expl-prob   Sets the probability of performing an exploration step for each variable. Can be ``normal`` or ``larger``. (default: ``normal``)
    --internal-logging   Whether to emit ``IDONE``'s internal logfiles. (default: false)

MVRSM
-----
:publications: :cite:`bliek2020black`
:bibtex:      ``bliek2020black``
:repository:   `GitHub <https://github.com/lbliek/MVRSM>`_
:supports:    ``cont``, ``int``, ``cat`` (as interpreted as integer / binarized).

    .. note:: 
        Note that behaviour differs (defaults to a fixed 1000 of basis functions) in the case that the function
        is only continuous.
    
:parameters:
    --rand-evals   The number of random evaluations to perform before utilizing the surrogate model. (default: 5)
    --model   The kind of model to use. Choose between ``basic`` and ``advanced``. (default: ``advanced``)
    --binarize-categorical   Whether to binarize categorical variables. Will turn a categorical variable with :math:`k` possible values, into :math:`\log_2(k)` binary (0 or 1) categorical variables. (default: false)
    --scaling   Whether to perform scaling based on the first sample. (default: false)

CoCaBO
------
:publications: :cite:`ru2019bayesian`
:bibtex:      ``ru2019bayesian``
:repository:   `GitHub <https://github.com/rubinxin/CoCaBO_code>`_
:supports:    ``cont``, ``int`` (interpreted as categorical), ``cat``. 

    Currently requires at least one continuous (``cont``) and one discrete (``int``, ``cat``) variable.
:parameters:
    --rand-evals   The number of random evaluations to perform before utilizing the surrogate model. (default: 24)

bayesianoptimization
--------------------
:bibtex:      ``bayesianoptimization``
:repository:   `GitHub <https://github.com/fmfn/BayesianOptimization>`_, `PyPI <https://pypi.org/project/bayesian-optimization/>`_
:supports:    ``cont``, ``int`` (via rounding), ``cat`` (interpreted as integer, via rounding)
:parameters:
    --rand-evals   The number of random evaluations to perform before utilizing the surrogate model. (default: 5)

pyGPGO
------
:publications: :cite:`Jimenez2017`
:bibtex:      ``pygpgo``, ``Jimenez2017``
:repository:   `GitHub <https://github.com/josejimenezluna/pyGPGO>`_, `PyPI <https://pypi.org/project/pyGPGO/>`_
:supports:      ``cont``, ``int`` (via rounding), ``cat`` (interpreted as integer, via rounding)
    
    .. note:: Built-in support for integers is not used due to crashes.
    
:parameters:
    --rand-evals   The number of random evaluations to perform before utilizing the surrogate model. (default: 3)
