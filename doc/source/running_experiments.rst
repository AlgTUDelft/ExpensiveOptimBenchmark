Running Experiments
===================

Running experiments is done through ``expensiveoptimbenchmark/run_experiment.py``. It is recommended to make use of the Singularity container, which installs the requirements for all approaches and problems. Python 3.6+ is required to run the benchmark suite.

``run_experiment.py``
---------------------

The general command for running an experiment is listed below, for information for the ``<problem>`` and ``<approach>`` specifiers, refer to :ref:`problems` and :ref:`approaches` respectively, and look at ``problem-key`` and ``approach-key``. The arguments directly following the name of the problem, are specifically for the problem itself. Similarly, the arguments directly following an approach form a configuration, and multiple configurations can be listed -- even those including the same approach.

.. code-block:: shell

    python run_experiment.py <general-arguments> <problem> <problem-args*> (<approach> <approach-args*>)+

For example:

.. code-block:: shell

    python ./run_experiment.py --max-eval=100 rosenbrock --n-cont=10 randomsearch hyperopt

To run an experiment with 100 evaluations on the 10-dimensional rosenbrock function, with all variables, using both random search and Hyperopt.

General arguments
#################

.. '--repetitions', '--max-eval', '--out-path', '--write-every', '--rand-evals-all'

--max-eval   Set the maximum number of evaluations.
--rand-evals-all   Set ``--rand-evals`` for all approaches. If not set, each approach uses their own default.
--repetitions   Number of times to repeat an experiment. (default: 1)
--out-path   The folder in which to place the experimental results.
    .. note:: In case of working with containers below, ensure the output directory is mounted to access the files after running an experiment. (default: ./results)
--write-every   The number of iterations in between writing calls to disk. If none, write after a full run (all evaluations) is completed. (default: none)

Docker and Singularity
----------------------
A subset of the problems -- ESP and PitzDaily -- are from :cite:`daniels2018suite` and the corresponding `BitBucket repository <https://bitbucket.org/arahat/cfd-test-problem-suite/>`__. The required installation to run these experiments makes use of OpenFOAM and is located in a Docker container or Singularity container.

Running these problems provides a few options:

1. Run everything inside the Singularity container:
    - Make sure Singularity 3.x is installed.
    - Build the container, for example by running ``buildCFD_singularity.sh``.
    - Run experiments
        .. code-block:: shell
    
            sudo singularity run --writable-tmpfs ./CFD.sif "python3.7 /home/openfoam/expensiveoptimbenchmark/run_experiment.py ..."

        .. tip:: 
            Singularity mounts the home directory of the current user by default. As such, writing the result files to ``~/results/`` or similar, will avoid loss of data.

        .. important:: 
            The CFD Benchmarks create new files, whereas the singularity container is by default read-only.
            ``--writable-tmpfs`` will use a small amount of RAM to write these temporary files to.
            Sadly, the writing permissions for this storage seem to be different from the folder itself,
            and the root user must be used in order to be able to write files. (Hence the usage of ``sudo`` in the command above)

2. Run everything inside the Docker container:
    - Make sure Docker is installed.
    - Build the container, for example by running ``buildCFD_docker.sh`` or ``buildCFD_docker.bat``.
    - Run the experiment:
        .. code-block:: shell
    
            sudo docker run cfdbench "python3.7 /home/openfoam/expensiveoptimbenchmark/run_experiment.py ..."

        .. important::
            Docker does not mount any directories of the host system by default. Make sure to include a ``--mount`` argument to the ``docker run`` command above for the result directory.

3. Run the CFD simulation inside the Docker container:
    .. important:: 

        This starts up a new Docker container for every evaluation, which may introduce additional overhead.

    - Make sure Docker is installed.
    - Make sure all the dependencies of approaches are properly installed. See :ref:`installation-notes` for potential issues.
      Otherwise poetry should take care of the dependencies for you.
    - Run directly on your machine itself, with admin rights if neccesary for docker (eg. on Mac and Linux).

    .. code-block:: shell
        python ./expensiveoptimbenchmark/run_experiment.py ...

.. _installation-notes:
Installation Notes
------------------
Usage of the containers is recommended, which provides all requirements for all approaches and problems.

We cover some potential issues with specific dependencies for some approaches.

- SMAC3 makes use of swig 3 in order to call its C++ code. If swig 4 is used, a segmentation fault will occur.
    - If changing to swig 3 and reinstalling does not work, make sure to clean up pip's cache.
- DONE(jl) is written in Julia, and therefore requires Julia to be installed including dependencies. (``Distributions``, ``NLopt``, ``PyCall``)
    - Special case for ``pyjulia`` (Python package) and ``pycall`` (Julia package) should be linked to the right Python version.
    - A workaround is present to deal with statically linked versions on Ubuntu in ``run_experiment.py``. See `this page in the pyjulia documentation <https://pyjulia.readthedocs.io/en/latest/troubleshooting.html#your-python-interpreter-is-statically-linked-to-libpython>`__. Expand the requirements to your needs if neccesary.