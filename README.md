# EXPObench:  An  EXPensive Optimization  benchmark  library

[**Extensive documentation is available here**](http://algtudelft.github.io/ExpensiveOptimBenchmark/)

<!-- TODO: Better name! -->

<!-- TODO: Insert general idea of why this benchmark is needed here -->

This repository is based on the paper [EXPObench: Benchmarking Surrogate-based Optimisation Algorithms on Expensive Black-box Functions](https://arxiv.org/abs/2106.04618).
The purpose of this repository is to benchmark different surrogate algorithms on expensive real life optimization problems. It contains different *problems*, where the goal is to minimize some objective function, and *approaches*, which solve the minimization problem through the use of surrogate models. The documentation above contains a list of the problems and approaches, as well as instructions for how to add new problems or approaches.

If you make use of EXPObench in your scientific work, please cite us:

```bibtex
@article{bliek2021expobench,
      title={{EXPObench}: Benchmarking Surrogate-based Optimisation Algorithms on Expensive Black-box Functions}, 
      author={Laurens Bliek and Arthur Guijt and Rickard Karlsson and Sicco Verwer and Mathijs de Weerdt},
      year={2021},
      eprint={2106.04618},
      primaryClass={cs.LG},
      journal={arXiv preprint arXiv:2106.04618}
}
```

This repository requires a working installation of python 3. A quick test to see if everything works is to run the following code. This runs random search (not a surrogate algorithm) on the rosenbrock problem (which is not an expensive optimization problem):

`python expensiveoptimbenchmark/run_experiment.py --max-eval=100 rosenbrock --n-cont=10 randomsearch`

The required packages for this minimal example are:
- numpy
- scipy
- hyperopt

All these packages can be installed using `pip install <package>`. Other problems and approaches in this repository may require additional packages.

The results of the minimal working example above will be put in the `results` folder. For more information, please refer to the [documentation](http://algtudelft.github.io/ExpensiveOptimBenchmark/).

## Docker container

Some of the problems, namely ESP and Pitzdaily, require the use of a Docker container. Information on how to run these problems can be found [in the documentation](https://algtudelft.github.io/ExpensiveOptimBenchmark/running_experiments.html).

## Poetry

If the use of Docker or Singularity is not desired, we recommend the use of Poetry to manage the packages that are required for the different problems and approaches.

- A virtualenv can be created through the use of [Poetry](https://github.com/python-poetry/poetry), which will automatically incorporate the neccesary dependencies:
    ```
    poetry install
    ```
- Once finished, you can open a shell in the virtualenv by running
    ```
    poetry shell
    ```
    or run a singular command using
    ```
    poetry run python ...
    ```
<!-- TODO: Once added, add method for running approach here -->


## Contact information

Besides the regular ways of contact on github, if you have any questions please contact `l.bliek@tue.nl`.
