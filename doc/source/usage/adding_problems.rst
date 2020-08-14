.. _adding-problems:

Adding Problems
===============

Adding a new benchmark problem consists of the following steps:

1. Define a problem class
2. Create a problem instance constructor
3. Add the problem to the ``problems`` dictionary in ``run_experiment.py``

Define a problem class
----------------------

A problem instance is defined to be a 'bundle' of an evaluation function, alongside a search space specification. This bundle is specified by subclassing ``BaseProblem``::

    from ..base import BaseProblem

    class <name>(BaseProblem):

        def __init__(self, ...):
            ...

        def evaluate(self, x):
            ...

        def lbs(self):
            ...

        def ubs(self):
            ...

        def vartype(self):
            ...

        def dims(self):
            ...

        def __str__(self):
            ...

``__init__(self)``
    The class initializer. Add instance parameters here.

``evaluate(self, x)``
    The evaluation function, ``x`` is a 1d numpy array of length ``dims``. Should return the objective value.

``lbs(self)`` and ``ubs(self)``
    Should both return vectors of length ``dims``, containing the lower and upper bounds of the problem respectively. Bounds are both inclusive.

``vartype(self)``
    Should return a vector of length ``dims`` containing the variable type of each variable. Value can be ``cont`` for continuous, ``int`` for integer, ``cat`` for categorical.

    Prefer ``int`` in case the order of values matters, otherwise use ``cat``.

``dims(self)``
    Return the number of dimensions.

Create a problem instance constructor
-------------------------------------

The parameters of a problem instance are passed as command line arguments. The processing of these arguments is performed by the instance constructor function.

The function has a fixed function signature, and returns a list of problem instances::

    def construct_<name>(params):
        ...
        return [...]

Returning a list allows parameters to define a set of instances, rather than only encode a single instance at a time.

.. tip:
    One can use ``parse_numerical_ranges`` to obtain a list of numbers from a string defining ranges. ``0:3,15`` for example results in a list ``[0, 1, 2, 3, 15]``.
    Multiples of these lists can be combined using itertools' ``product``.

Adding the problem to ``run_experiment.py``
-------------------------------------------

Finally the problem can be added to the command line parser in ``run_experiment.py`` by adding the following entry to the ``problems`` dictionary::

    '<problem-name>': {
        'args': {'--<arg-0>', '--<arg-1>', ...}
        'defaults': {
            '--<arg-0>': '100',
            ...
        },
        'constructor': construct_<name>
    }

With this entry added, the following should run an experiment with the newly added problem:

.. code-block:: bash

    python ./expensiveoptimbenchmark/run_experiment.py <global config> <problem-name> <problem-args> <approach-0-name> <approach-0-args> <approach-1-name> <approach-1-args>



.. tip::
    Make sure to run a small experiment first to verify everything works as expected!