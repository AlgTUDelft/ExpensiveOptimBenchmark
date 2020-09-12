Running Experiments
===================

Running experiments is done through ``expensiveoptimbenchmark/run_experiment.py``. It is recommended to make use of the Singularity container, which installs the requirements for all approaches and problems. Python 3.6+ is required to run the benchmark suite.

``run_experiment.py``
---------------------

The general command for running an experiment is listed below, for information for the ``<problem>`` and ``<approach>`` specifiers, refer to :ref:`problems` and :ref:`approaches` respectively, and look at ``problem-key`` and ``approach-key``. 

.. code-block:: shell

    python run_experiment.py <general-arguments> <problem> <problem-args*> (<approach> <approach-args*>)+

For example:

.. code-block:: shell

    python ./run_experiment.py --max-eval=100 rosenbrock --n-cont=10 randomsearch hyperopt

To run an experiment with 100 evaluations on the 10-dimensional rosenbrock function, with all variables, using both random search and Hyperopt.

General arguments
#################

--max-eval   Set the maximum number of evaluations.
--all-max-rand-evals   Set ``--rand-evals`` for all approaches. If not set, each approach uses their own default.
--repetitions   Number of times to repeat an experiment.
