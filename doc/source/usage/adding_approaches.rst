.. _adding-approaches:

Adding Approaches
=================

Adding a new approch to the benchmark suite consists of a few steps:

1. Writing a wrapper. Which works as an adaptor for dealing with approach-specific pecularities.
2. Writing an executor. Validates and processes command line arguments and passes them on to the wrapper for performing an experimental run of the approach on a problem.
3. Add the approach to the ``solvers`` dictionary.

Creating the wrapper
--------------------

The wrapper is often placed under ``expensiveoptimbenchmark/solvers/<name>/w<name>.py`` and acts as an adaptor between the approach itself and the benchmark suite:

1. Convert the problem specification to the input of the approach.
2. Attach the monitor to objective function
3. Run the approach on the problem
4. Return the solution and the monitor.

For examples refer to the already wrapped approaches under ``expensiveoptimbenchmark/solvers/``.

A problem specifies 5 methods:

``dims``
    An integer specifying the number of dimensions.    

``lbs`` and ``ubs``
    An array of length ``dims()``, specifying the lower and upper bound of each variable respectively

``vartype``
    An array of length ``dims()``, specifying whether a variable is continuous (``cont``), integer (``int``) or categorical (``cat``). A variable is generally chosen to be integer when order matters, and otherwise categorical.

``dependencies``
    An array of length ``dims()``, which is ``None`` if the variable at this position does not have any dependencies, and is otherwise a dictionary containing ``on`` -- specifying the variable index that the variable is depndent on -- as well as ``values`` specifying the set of values that cause a variable to be active.

.. note::
    How to specify a problem is documented under :ref:`adding-problems`.

The monitor is a particular tool to keep track of evaluations performed and time spent on both deciding on a point (referred to as ``model_time`` in the output csv file) and the time spent on evaluating this point (``eval_time``).

The class ``Monitor`` is located in ``expensiveoptimbenchmark/solvers/utils.py`` and can easily be imported via::

   from ..utils import Monitor

Constructing a ``Monitor`` requires a string as argument, specifying the name of the approach. Additionally, this string should include additional information about the parameterization of the approach::

    mon = Monitor("<name>/fast")

The monitor has a set of four markers: ``begin``, ``end``, ``commit_start_eval`` and ``commit_end_eval``. The first two -- ``begin`` and ``end`` -- are used to mark the beginning and start of the full run of an approach, whereas the last two -- ``commit_start_eval`` and ``commit_end_eval`` -- are used to mark the beginning and end of an evaluation.

A general layout of the wrapper with an optimizer which accepts a function is as follows::

    def get_variables(problem):
        # ... approach specific code here ...

    def optimize_optimizer(problem, max_evals, rand_evals, ..., log=None):

        # ... approach specific code here ...
        # spec = get_variables(problem)

        def f(x_in):
            # ... approach specific code here ...
            # eg. x = x_in
            mon.commit_start_eval()
            r = float(problem.evaluate(x))
            mon.commit_end_eval(x, r)
            return r

        # ... approach specific code here ...
        # eg. optimizer = Optimizer(f, spec)

        mon.start()
        # ... approach specific code here ...
        # eg. optimizer.optimize()
        mon.end()

        # ... approach specific code here ...
        # eg. return optimizer.best_x, optimizer.best_obj, mon


Creating the executor and adding the approach
---------------------------------------------
The executor is responsible for parsing the command line arguments specific to an approach, and running the approach with these parameters. An executor has a fixed function signature::

    def execute_<name>(params, problem, max_eval, log):
        from solvers.<name>.w<name> import optimize_<name>
        # ... process params ...
        return  optimize_<name>(problem, max_eval, 

Special attention has been given to the number of initial random evaluations of an objective function, as such it is recommended to use ``--rand-evals`` for this value. If ``--rand-evals-all`` is set, ``--rand-evals`` will be set to this value among all approaches as default, overriding the approach specific default.

Finally the approach can be added to the solvers dictionary in ``run_experiment.py``. An entry of this dictionary has the following format::

    '<name>': {
        'args': {'--rand-evals', ...},
        'defaults': {
            '--rand-evals': '5',
            ...
        },
        'executor': execute_<name>,
        'check': nop # or check_<name>
    }

**args** 
    A ``set`` of command line arguments. 

**defaults**
    A ``dict`` of default values for command line arguments.

**executor**
    The executor function you have defined previously.
    
    Called for every repetition of an approach.

**check**
    The check function. If you want to run any checks before starting the experiment, a check function can be provided. One can for example verify that the parameter values are valid, or that all required packages are installed and present.
    
    Called once before starting all experimental runs.


With all this done, your approach should be runnable with:

.. code-block:: bash

    python ./expensiveoptimbenchmark/run_experiment.py <global config> <problem-name> <problem-args> <approach-0-name> <approach-0-args> <approach-1-name> <approach-1-args>

.. tip::
    Make sure to run a small experiment first to verify everything works as expected!