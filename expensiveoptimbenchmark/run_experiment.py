import sys

# Specialized argument processing for problems
# TSP
from problems.TSP import TSP, load_tsplib, load_explicit_tsp
def construct_tsp(params):
    iters = int(params['--iter'])
    if '--tsplib-file' in params:
        return [load_tsplib(params['--tsplib-file'], iters)]
    elif '--explicit-file' in params:
        return [load_explicit_tsp(params['--explicit-file'], iters)]
    else:
        raise ValueError('No instance file provided for TSP. Specify one using `--tsplib-file` or `--explicit-file`')

# Convex
from problems.convex import Convex
def construct_convex(params):
    d = int(params['-d'])
    seed = int(params['--seed'])
    return [Convex(d, seed)]

# IntRosenbrock
from problems.rosenbrock_int import RosenbrockInt
def construct_rosen(params):
    d = int(params['-d'])
    return [RosenbrockInt(d)]

# Summary of problems and their parameters.
problems = {
    'tsp': {
        'args': {'--tsplib-file', '--explicit-file', '--iter'},
        'defaults': {
            '--iter': '100'
        },
        'constructor': construct_tsp
    },
    'rosen': {
        'args': {'-d'},
        'defaults': {
            '-d': '2'
        },
        'constructor': construct_rosen
    },
    'convex': {
        'args': {'--seed', '-d'},
        'defaults': {
            '--seed': '0'
        },
        'constructor': construct_convex
    }
}

## IDONE
from solvers.IDONE.wIDONE import optimize_IDONE
def execute_IDONE(params, problem, max_eval):
    if params['--model'] not in ['basic', 'advanced']:
        raise ValueError("Valid model types are `basic` and `advanced`")
        
    type_model = params['--model']

    return optimize_IDONE(problem, max_eval, model=type_model)

# Hyperopt TPE
from solvers.hyperopt.whyperopt import optimize_hyperopt_tpe
def execute_hyperopt(params, problem, max_eval):
    # TODO: Set number of random evaluations?
    return optimize_hyperopt_tpe(problem, max_eval)

# pyGPGO
from solvers.pyGPGO.wpyGPGO import optimize_pyGPGO
from pyGPGO.covfunc import matern32
from pyGPGO.acquisition import Acquisition
from pyGPGO.surrogates.GaussianProcess import GaussianProcess

def execute_pygpgo(params, problem, max_eval):
    # TODO: Allow picking different values for these?
    cov = matern32()
    gp = GaussianProcess(cov, optimize=True, usegrads=True)
    acq = Acquisition(mode='ExpectedImprovement')
    return optimize_pyGPGO(problem, max_eval, gp, acq)

solvers = {
    'idone': {
        'args': {'--model'},
        'defaults': {
            '--model': 'advanced'
        },
        'executor': execute_IDONE
    },
    'hyperopt': {
        'args': set(),
        'defaults': {
        },
        'executor': execute_hyperopt
    },
    'pygpgo': {
        'args': {'--acquisition'},
        'defaults': {
        },
        'executor': execute_pygpgo
    },
    'bayesianoptimization': {
        'args': set(),
        'defaults': {
        },
        'executor': None #TODO!
    }
}

general_args = {'--repetitions', '--max-eval'}

# Parse
general = {
    '--repetitions': 1
}
problem = {}
solver = {}


i = 1
args = sys.argv

if len(args) == 1 or (len(args) == 2 and (args[1] == '-h' or args[1] == '--help')) :
    print(f"Arguments: [general args] [problem] [problem args] [solver] [solver args]")
    print(f"General args:")
    print(f" --repetitions=<int> \t Set the number of repetitions of the experiment (default: 1)")
    print(f" Note: if a problem is specified in a way that it produces multiples")
    print(f"       each of these problems are repeated `repetitions` times.")
    print(f" --max-eval=<int> \t Set the maximum number of evaluations (required)")
    print()
    print(f"Problems:")
    print()
    # TSP
    print(f" tsp")
    print(f" --tsplib-file=<path> \t\t Parse and load a TSPLIB instance")
    print(f" --explicit-file=<path> \t Parse and load an explicit TSP instance with filename as name and an explicit W as content")
    print(f" --iter=<int> \t\t\t Number of noisy TSP evaluations (default: 100)")
    print(f" (Providing one of `--tsplib-file` or `--explicit-file` is required.) ")
    print()
    # Convex
    print(f" convex")
    print(f" --seed=<int> \t The seed of the convex problem instance (default: 0)")
    print(f" -d=<int> \t The dimensionality of the convex problem instance")
    print()
    # Rosenbrock
    print(f" rosen")
    print(f" -d=<int> \t The dimensionality of the rosenbrock problem")
    print()
    print(f"Solvers:")
    # IDONE
    print(f" idone")
    print(f" --model=<basic|advanced> \t The kind of model IDONE should utilize (default: advanced)")
    print()
    # HyperOpt
    print(f" hyperopt")
    print(f" (no arguments implemented yet)")
    print()
    # pyGPGO
    print(f" pygpgo")
    print(f" (no arguments implemented yet)")
    print()
    # bayesianoptimization
    print(f" bayesianoptimization")
    print(f" (no arguments implemented yet)")
    print()
    sys.exit(0)

while len(args) > i and args[i].startswith("-"):
    name_value = args[i].split("=")
    if name_value[0] not in general_args:
        print(f"Did not expect argument {name_value[0]}. Did you mean one of {general_args}?")
        sys.exit(-1)
    general[name_value[0]] = name_value[1]
    i += 1

repetitions = int(general['--repetitions'])
max_eval = int(general['--max-eval'])

if args[i] not in problems:
    print(f"Expected a problem. Possible options: {problems.keys()}.")
    sys.exit(-1)
problem['name'] = args[i]
problem['info'] = problems[args[i]]
problem['params'] = problem['info']['defaults']

i += 1
while len(args) > i and args[i].startswith("-"):
    name_value = args[i].split("=")
    if name_value[0] not in problem['info']['args']:
        print(f"Problem {problem['name']} does not accept argument {name_value[0]}")
        sys.exit(-1)
    problem['params'][name_value[0]] = name_value[1]
    i += 1

if len(args) <= i or args[i] not in solvers:
    print(f"Expected a solver. Possible options: {solvers.keys()}.")
    sys.exit(-1)

solver['name'] = args[i]
solver['info'] = solvers[args[i]]
solver['params'] = solver['info']['defaults']

while len(args) > i and args[i].startswith("-"):
    name_value = args[i].split("=")
    if name_value[0] not in problem['info']['args']:
        print(f"Problem {problem['name']} does not accept argument {name_value[0]}")
        sys.exit(-1)
    problem['params'][name_value[0]] = name_value[1]
    i += 1

# Actually perform the experiment.
problems = problem['info']['constructor'](problem['params'])

for r in range(repetitions):
    for problem_instance in problems:
        solY, solX, monitor = solver['info']['executor'](solver['params'], problem_instance, max_eval)