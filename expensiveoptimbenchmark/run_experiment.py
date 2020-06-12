import sys

from itertools import product

def parse_numerical_range(s):
    range_ = s.split(":")
    if len(range_) == 1:
        return [int(range_[0])]
    else:
        return range(int(range_[0]), int(range_[1])+1)

def parse_numerical_ranges(ranges):
    # Numerical ranges are defined to be
    # <ranges> = <range>|<range>,<ranges>
    # <range> = <a:n>|<a:n>:<b:n>
    # A range produces either a, or all numbers between a and b, including b.
    # Ranges generates the concatenation of each range contained within.
    return [int(i) for range_ in ranges.split(",") for i in parse_numerical_range(range_)]

# Specialized argument processing for problems
# TSP
def construct_tsp(params):
    from problems.TSP import TSP, load_tsplib, load_explicit_tsp
    iter_ns = parse_numerical_ranges(params['--iter'])
    if '--tsplib-file' in params:
        return [load_tsplib(params['--tsplib-file'], iters) for iters in iter_ns]
    elif '--explicit-file' in params:
        return [load_explicit_tsp(params['--explicit-file'], iters) for iters in iter_ns]
    else:
        raise ValueError('No instance file provided for TSP. Specify one using `--tsplib-file` or `--explicit-file`')

# Convex
def construct_convex(params):
    from problems.convex import Convex
    ds = parse_numerical_ranges(params['-d'])
    seeds = parse_numerical_ranges(params['--seed'])
    return [Convex(d, seed) for d, seed in product(ds, seeds)]

# IntRosenbrock
def construct_rosen(params):
    from problems.rosenbrock_int import RosenbrockInt
    ds = parse_numerical_ranges(params['-d'])
    return [RosenbrockInt(d) for d in ds]

# Linear MIVABO Function
def construct_linearmivabo(params):
    from problems.linear_MIVABOfunction import Linear
    return [Linear()]

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
    },
    'linearmivabo': {
        'args': set(), # TODO: make this approach configurable.
        'defaults': {
        },
        'constructor': construct_linearmivabo
    },
}

def generate_construct_synthetic(fn):
    def generate_synthetic(args):
        return [fn]
    return generate_synthetic

# Add syntheticfunctions.
from problems.syntheticfunctions import fns
problems.update({
    fn.name: {
        'args': set(),
        'defaults': {},
        'constructor': generate_construct_synthetic(fn)
    }
    for fn in fns
})

def nop(*x, **y):
    pass

## IDONE
def execute_IDONE(params, problem, max_eval, log):
    from solvers.IDONE.wIDONE import optimize_IDONE
    if params['--model'] not in ['basic', 'advanced']:
        raise ValueError("Valid model types are `basic` and `advanced`")
        
    type_model = params['--model']

    return optimize_IDONE(problem, max_eval, model=type_model, log=log)

def execute_MVRSM(params, problem, max_eval, log):
    from solvers.MVRSM.wMVRSM import optimize_MVRSM
    if params['--model'] not in ['basic', 'advanced']:
        raise ValueError("Valid model types are `basic` and `advanced`")
        
    type_model = params['--model']

    return optimize_MVRSM(problem, max_eval, model=type_model, log=log)


# Hyperopt TPE
def execute_hyperopt(params, problem, max_eval, log):
    from solvers.hyperopt.whyperopt import optimize_hyperopt_tpe
    # TODO: Set number of random evaluations?

    conversion_params = {
        'int_conversion_mode': params.get('--int-conversion-mode')
    }

    return optimize_hyperopt_tpe(problem, max_eval, cparams=conversion_params, log=log)

def execute_hyperopt_rnd(params, problem, max_eval, log):
    from solvers.hyperopt.whyperopt import optimize_hyperopt_rnd
    # TODO: Set number of random evaluations?
    return optimize_hyperopt_rnd(problem, max_eval, log=log)

# pyGPGO
def execute_pygpgo(params, problem, max_eval, log):
    from solvers.pyGPGO.wpyGPGO import optimize_pyGPGO
    from pyGPGO.covfunc import matern32
    from pyGPGO.acquisition import Acquisition
    from pyGPGO.surrogates.GaussianProcess import GaussianProcess
    # TODO: Allow picking different values for these?
    cov = matern32()
    gp = GaussianProcess(cov, optimize=True, usegrads=True)
    acq = Acquisition(mode='ExpectedImprovement')
    return optimize_pyGPGO(problem, max_eval, gp, acq, log=log)

# bayesian-optimization
def execute_bayesianoptimization(params, problem, max_eval, log):
    from solvers.bayesianoptimization.wbayesianoptimization import optimize_bayesian_optimization
    # TODO: Allow picking different configurations?
    return optimize_bayesian_optimization(problem, max_eval, log=log)

# smac
def execute_smac(params, problem, max_eval, log):
    from solvers.smac.wsmac import optimize_smac
    return optimize_smac(problem, max_eval, log=log)

def check_smac():
    from solvers.smac.wsmac import optimize_smac
    pass

# CoCaBO
def execute_cocabo(params, problem, max_eval, log):
    from solvers.CoCaBO.wCoCaBo import optimize_CoCaBO
    return optimize_CoCaBO(problem, max_eval, log=log)

solvers = {
    'idone': {
        'args': {'--model'},
        'defaults': {
            '--model': 'advanced'
        },
        'executor': execute_IDONE,
        'check': nop
    },
    'mvrsm': {
        'args': {'--model'},
        'defaults': {
            '--model': 'advanced'
        },
        'executor': execute_MVRSM,
        'check': nop
    },
    'hyperopt': {
        'args': {'--int-conversion-mode'},
        'defaults': {
            '--int-conversion-mode': 'quniform'
        },
        'executor': execute_hyperopt,
        'check': nop
    },
    'randomsearch': {
        'args': set(),
        'defaults': {
        },
        'executor': execute_hyperopt_rnd,
        'check': nop
    },
    'pygpgo': {
        'args': {'--acquisition'},
        'defaults': {
        },
        'executor': execute_pygpgo,
        'check': nop
    },
    'bayesianoptimization': {
        'args': set(),
        'defaults': {
        },
        'executor': execute_bayesianoptimization,
        'check': nop
    },
    'smac': {
        'args': set(),
        'defaults': {
        },
        'executor': execute_smac,
        'check': check_smac
    },
    'cocabo': {
        'args': set(),
        'defaults': {
        },
        'executor': execute_cocabo,
        'check': nop
    }
}

general_args = {'--repetitions', '--max-eval', '--out-path', '--write-every'}

# Parse
general = {
    '--repetitions': 1,
    '--out-path': './results/',
    '--write-every': 'none',
}
problem = {}
solver = {}
current_solvers = []


i = 1
args = sys.argv

if len(args) == 1 or (len(args) == 2 and (args[1] == '-h' or args[1] == '--help')) :
    print(f"Arguments: [general args] [problem] [problem args] [solver] [solver args]")
    print(f"General args:")
    print(f" --repetitions=<int> \t Set the number of repetitions of the experiment (default: 1)")
    print(f" Note: if a problem is specified in a way that it produces multiples")
    print(f"       each of these problems are repeated `repetitions` times.")
    print(f" --max-eval=<int> \t Set the maximum number of evaluations (required)")
    print(f" --out-path=<path> \t Where to place the logfiles. (default: ./results)")
    print(f" --write-every=<int|none> \t Update logfiles every int iterations (default: none)")
    print(f" Note: none indicates that the logfile will only be updated once an approach")
    print(f"       has exhausted its maximum number of evaluations.")
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
    print(f" --seed=<intranges> \t The seed of the convex problem instance (default: 0)")
    print(f" -d=<intranges> \t The dimensionality of the convex problem instance")
    print()
    # Rosenbrock
    print(f" rosen")
    print(f" -d=<intranges> \t The dimensionality of the rosenbrock problem")
    print()
    print(f"Solvers:")
    # IDONE
    print(f" idone")
    print(f" --model=<basic|advanced> \t The kind of model IDONE should utilize (default: advanced)")
    print()
    # MVRSM
    print(f" mvrsm")
    print(f" --model=<basic|advanced> \t The kind of model MVRSM should utilize (default: advanced)")
    print()
    # HyperOpt
    print(f" hyperopt")
    print(f" --int-conversion-mode=<quniform|randint> \t The default conversion mode for integers for hyperopt (default: quniform)")
    print()
    # HyperOpt / randomsearch
    print(f" randomsearch")
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
    # smac
    print(f" smac")
    print(f" (no arguments implemented yet)")
    print()
    # CoCaBO
    print(f" cocabo")
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
out_path = general['--out-path']
write_every = None if general['--write-every'] == "none" else int(general['--write-every'])

if write_every is not None and write_every <= 0:
    print(f"`--write-every should have a value > 1.")
    sys.exit(-1)

if out_path[-1] != '/':
    out_path = out_path + '/'

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

while len(args) > i:
    if len(args) <= i or args[i] not in solvers:
        print(f"Expected a solver. Possible options: {solvers.keys()}.")
        sys.exit(-1)

    solver['name'] = args[i]
    solver['info'] = solvers[args[i]]
    solver['params'] = solver['info']['defaults'].copy()

    # Perform imports before running so that we do not run
    # into surprises later
    try:
        solver['info']['check']()
    except:
        print(f"Dependencies for {solver['name']} seem to be missing.")
        print(f"Did you install the relevant extras?")
        sys.exit(-1)
    i += 1

    while len(args) > i and args[i].startswith("-"):
        name_value = args[i].split("=")
        if name_value[0] not in solver['info']['args']:
            print(f"Problem {solver['name']} does not accept argument {name_value[0]}")
            sys.exit(-1)
        solver['params'][name_value[0]] = name_value[1]
        i += 1
    
    current_solvers.append(solver.copy())
    solver = {}

if len(current_solvers) == 0:
        print(f"Expected a solver. Possible options: {solvers.keys()}.")
        sys.exit(-1)

## Actually perform the experiment.
import os
import time
import random

problems = problem['info']['constructor'](problem['params'])

os.makedirs(out_path, exist_ok=True)

t = time.time()
rnd = random.randint(1, 1<<14)
logfile_iters = f"{out_path}experiment_{problem['name']}_{t}_{rnd}_iters.csv"
logfile_summary = f"{out_path}experiment_{problem['name']}_{t}_{rnd}_summ.csv"

loginfo = {
    'file_iters': logfile_iters,
    'file_summary': logfile_summary,
    'write_every': write_every,
    'emit_header': True
}

for solver in current_solvers:
    for problem_instance in problems:
        for r in range(repetitions):
            solY, solX, monitor = solver['info']['executor'](solver['params'], problem_instance, max_eval, log=loginfo)
            with open(logfile_iters, 'a') as f:
                from_iter = 0
                if write_every is not None:
                    from_iter = monitor.num_iters - (monitor.num_iters % write_every)
                monitor.emit_csv_iters(f, from_iter=from_iter, emit_header=loginfo['emit_header'])
            with open(logfile_summary, 'a') as f:
                monitor.emit_csv_summary(f, emit_header=loginfo['emit_header'])
            loginfo['emit_header'] = False