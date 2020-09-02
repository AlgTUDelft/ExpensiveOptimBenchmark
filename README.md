# ExpensiveOptimBenchmark

[Documentation](http://algtudelft.github.io/ExpensiveOptimBenchmark/)

<!-- TODO: Better name! -->

<!-- TODO: Insert general idea of why this benchmark is needed here -->

## Usage

Running this benchmark requires a working installation of python 3. 

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

# List of commands used

## Synthetic

Note: These experiments all were ran in singularity on the server. But can be ran using poetry as well, as the container environment is not neccesary (all dependencies are listed).
Running the command can be done inside the container using:
`singularity run CFD.sif "<command>"`

**Func3C**

```bash
python3.7 ./expensiveoptimbenchmark/run_experiment.py --repetitions=100 --out-path=./results/func3c/ --max-eval=224 --rand-evals-all=24 func3C randomsearch hyperopt mvrsm smac --deterministic=y cocabo bayesianoptimization
```

**Rosenbrock 10 (executed 4x in parellel)**

```bash
python3.7 ./expensiveoptimbenchmark/run_experiment.py --repetitions=25 --out-path=./results/rosen10/ --max-eval=124 --rand-evals-all=24 dim10Rosenbrock randomsearch hyperopt mvrsm smac --deterministic=y cocabo
```

**Linear MiVaBO (default) (executed 7x in parellel)**

```bash
python3.7 ./expensiveoptimbenchmark/run_experiment.py --repetitions=1 --out-path=./results/linearmivabo/  --max-eval=224 --rand-evals-all=24 linearmivabo --seed=111:118 --noisy=y --laplace=n randomsearch hyperopt mvrsm smac
```

**Linear MiVaBO (large) (executed 7x in parellel)**

```bash
python3.7 ./expensiveoptimbenchmark/run_experiment.py --repetitions=1 --out-path=./results/linearmivabo-large/ --max-eval=2024 --rand-evals-all=24 linearmivabo --dd=119 --dc=119 --seed=111 --noisy=y --laplace=n randomsearch hyperopt mvrsm smac
```

**Ackley 53 (executed 7x in parellel)**

```bash
python3.7 ./expensiveoptimbenchmark/run_experiment.py --repetitions=1 --out-path=./results/ackley53/ --max-eval=1024 --rand-evals-all=24 dim53Ackley randomsearch hyperopt smac --deterministic=y mvrsm cocabo bayesianoptimization
```

## Realistic

**ESP** **(executed 5x in parellel)** | [Problem Source](https://bitbucket.org/arahat/cfd-test-problem-suite/src/master/)

This problem requires the script to be ran inside the container environment (either Docker or Singularity), or docker to be installed and usable (eg. ``sudo python ...`` is used, such that ``docker run`` can be ran). The use of sudo is due to filesystem write permissions (if this can be fixed, sudo is no longer required), and leads to the end results being placed elsewhere.

```bash
sudo singularity run --writable-tmpfs CFD.sif "python3.7 /home/openfoam/expensiveoptimbenchmark/run_experiment.py --repetitions=1 --out-path=./results/esp/ --max-eval=100 --rand-evals-all=24 esp randomsearch hyperopt smac --deterministic=y mvrsm cocabo bayesianoptimization"
```

**Hyperparameter Optimization** **(executed 5x in parellel)**

Requires external data found at [https://archive.ics.uci.edu/ml/datasets/Steel+Plates+Faults](https://archive.ics.uci.edu/ml/datasets/Steel+Plates+Faults), and was unpacked in `./data/fault`

```bash
python3.7 ./expensiveoptimbenchmark/run_experiment.py --repetitions=2 --out-path=./results/bo/esp3/ --max-eval=224 --rand-evals-all=24 automl --folder=./data/fault randomsearch hyperopt smac --deterministic=y mvrsm cocabo bayesianoptimization
```