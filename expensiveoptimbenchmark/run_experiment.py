import os
import sys

from itertools import product

# Stop numpy and scipy from doing multithreading automatically.
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

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
    noise_seeds = parse_numerical_ranges(params['--noise-seed']) if params['--noise-seed'] != "random" else [None]
    if '--tsplib-file' in params:
        return [load_tsplib(params['--tsplib-file'], iters, noise_seed) for (iters, noise_seed) in product(iter_ns, noise_seeds)]
    elif '--explicit-file' in params:
        return [load_explicit_tsp(params['--explicit-file'], iters, noise_seed) for (iters, noise_seed) in product(iter_ns, noise_seeds)]
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
    logscale = params['--logscale'] in ['true','t', 'yes', 'y']
    return [RosenbrockInt(d, logscale) for d in ds]

# Linear MIVABO Function
def construct_linearmivabo(params):
    from problems.linear_MIVABOfunction import Linear

    maybe_seed = params.get('--seed')
    seeds = parse_numerical_ranges(maybe_seed) if maybe_seed is not None else [None]
    laplace = params['--laplace'] in ['true','t', 'yes', 'y']
    noisy = params['--noisy'] in ['true','t', 'yes', 'y']

    return [Linear(noisy=noisy, laplace=laplace, seed=seed) for seed in seeds]

# floris wake simulator
def construct_windwake(params):
    from problems.windwake import WindWakeLayout
    sim_info_file = params['--file']
    wind_seed = int(params['--wind-seed'])
    n_turbines = int(params['-n'])
    width = int(params['-w'])
    height = int(params['-h'])

    return [WindWakeLayout(sim_info_file, n_turbines=n_turbines, wind_seed=wind_seed, width=width, height=height)]

def construct_windwakeh(params):
    from problems.windwakeheight import WindWakeHeightLayout
    sim_info_file = params['--file']
    wind_seed = int(params['--wind-seed'])
    n_turbines = int(params['-n'])
    width = int(params['-w'])
    height = int(params['-h'])

    return [WindWakeHeightLayout(sim_info_file, n_turbines=n_turbines, wind_seed=wind_seed, width=width, height=height)]

# Summary of problems and their parameters.
problems = {
    'tsp': {
        'args': {'--tsplib-file', '--explicit-file', '--iter', '--noise-seed'},
        'defaults': {
            '--iter': '100',
            '--noise-seed': '0'
        },
        'constructor': construct_tsp
    },
    'rosen': {
        'args': {'-d', '--logscale'},
        'defaults': {
            '-d': '2',
            '--logscale': 'f'
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
        'args': {'--seed', '--laplace', '--noisy'}, # TODO: make this approach configurable.
        'defaults': {
            '--laplace': 'y',
            '--noisy': 'n'
        },
        'constructor': construct_linearmivabo
    },
    'windwake': {
        'args': {'--file', '-n', '-w', '-h', '--wind-seed'},
        'defaults': {
            '-n': '3',
            '-w': '1000',
            '-h': '1000',
            '--wind-seed': '0'
        },
        'constructor': construct_windwake
    },
    'windwakeh': {
        'args': {'--file', '-n', '-w', '-h', '--wind-seed'},
        'defaults': {
            '-n': '3',
            '-w': '1000',
            '-h': '1000',
            '--wind-seed': '0'
        },
        'constructor': construct_windwakeh
    }
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

from problems.DockerCFDBenchmark import dockersimbenches
problems.update({
    dockerproblem.name.lower(): {
        'args': set(),
        'defaults': {},
        'constructor': generate_construct_synthetic(dockerproblem)
    }
    for dockerproblem in dockersimbenches
})

def nop(*x, **y):
    pass

## IDONE
def execute_IDONE(params, problem, max_eval, log):
    from solvers.IDONE.wIDONE import optimize_IDONE
    if params['--model'] not in ['basic', 'advanced']:
        raise ValueError("Valid model types are `basic` and `advanced`")
    if params['--binarize-categorical'] not in ['true', 't', 'yes', 'y', 'false', 'f', 'no', 'n']:
        raise ValueError("--binarize-categorical should be a boolean.")
    if params['--scaling'] not in ['true', 't', 'yes', 'y', 'false', 'f', 'no', 'n']:
        raise ValueError("--scaling should be a boolean.")
    if params['--internal-logging'] not in ['true', 't', 'yes', 'y', 'false', 'f', 'no', 'n']:
        raise ValueError("--internal-logging should be a boolean.")
        
    type_model = params['--model']
    binarize_categorical = params['--binarize-categorical'] in ['true','t', 'yes', 'y']
    enable_scaling = params['--scaling'] in ['true','t', 'yes', 'y']
    idone_log = params['--internal-logging'] in ['true','t', 'yes', 'y']
    rand_evals = int(params['--rand-evals']) - 1
    assert rand_evals >= 0, "IDONE requires at least one initial random evaluation."

    return optimize_IDONE(problem, max_eval, rand_evals=rand_evals, model=type_model, binarize_categorical=binarize_categorical, enable_scaling=enable_scaling, log=log, idone_log=idone_log)

def execute_MVRSM(params, problem, max_eval, log):
    from solvers.MVRSM.wMVRSM import optimize_MVRSM
    if params['--model'] not in ['basic', 'advanced']:
        raise ValueError("Valid model types are `basic` and `advanced`")
    if params['--binarize-categorical'] not in ['true', 't', 'yes', 'y', 'false', 'f', 'no', 'n']:
        raise ValueError("--binarize-categorical should be a boolean.")
    if params['--scaling'] not in ['true', 't', 'yes', 'y', 'false', 'f', 'no', 'n']:
        raise ValueError("--scaling should be a boolean.")

    type_model = params['--model']
    binarize_categorical = params['--binarize-categorical'] in ['true','t', 'yes', 'y']
    enable_scaling = params['--scaling'] in ['true','t', 'yes', 'y']
    rand_evals = int(params['--rand-evals']) - 1
    assert rand_evals >= 0, "MVRSM requires at least one initial random evaluation."

    return optimize_MVRSM(problem, max_eval, rand_evals=rand_evals, model=type_model, binarize_categorical=binarize_categorical, enable_scaling=enable_scaling, log=log)

# SA
def execute_SA(params, problem, max_eval, log):
    from solvers.SA.wSA import optimize_SA

    return optimize_SA(problem, max_eval, log=log)


# Hyperopt TPE
def execute_hyperopt(params, problem, max_eval, log):
    from solvers.hyperopt.whyperopt import optimize_hyperopt_tpe
    rand_evals = int(params['--rand-evals'])

    conversion_params = {
        'int_conversion_mode': params.get('--int-conversion-mode')
    }

    return optimize_hyperopt_tpe(problem, max_eval, random_init_evals=rand_evals, cparams=conversion_params, log=log)

def execute_hyperopt_rnd(params, problem, max_eval, log):
    from solvers.hyperopt.whyperopt import optimize_hyperopt_rnd

    conversion_params = {
        'int_conversion_mode': params.get('--int-conversion-mode')
    }

    return optimize_hyperopt_rnd(problem, max_eval, cparams=conversion_params, log=log)

# pyGPGO
def execute_pygpgo(params, problem, max_eval, log):
    from solvers.pyGPGO.wpyGPGO import optimize_pyGPGO
    from pyGPGO.covfunc import matern32
    from pyGPGO.acquisition import Acquisition
    from pyGPGO.surrogates.GaussianProcess import GaussianProcess
    rand_evals = int(params['--rand-evals'])

    # TODO: Allow picking different values for these?
    cov = matern32()
    gp = GaussianProcess(cov, optimize=True, usegrads=True)
    acq = Acquisition(mode='ExpectedImprovement')
    return optimize_pyGPGO(problem, max_eval, gp, acq, random_init_evals=rand_evals, log=log)

# bayesian-optimization
def execute_bayesianoptimization(params, problem, max_eval, log):
    from solvers.bayesianoptimization.wbayesianoptimization import optimize_bayesian_optimization
    rand_evals = int(params['--rand-evals'])
    # TODO: Allow picking different configurations?
    return optimize_bayesian_optimization(problem, max_eval, random_init_evals=rand_evals, log=log)

# smac
def execute_smac(params, problem, max_eval, log):
    from solvers.smac.wsmac import optimize_smac
    rand_evals = int(params['--rand-evals'])
    deterministic = params.get('--deterministic') in ['true','t', 'yes', 'y']
    return optimize_smac(problem, max_eval, rand_evals=rand_evals, deterministic=deterministic, log=log)

def check_smac():
    from solvers.smac.wsmac import optimize_smac
    pass

# CoCaBO
def execute_cocabo(params, problem, max_eval, log):
    from solvers.CoCaBO.wCoCaBo import optimize_CoCaBO
    rand_evals = int(params['--rand-evals'])
    return optimize_CoCaBO(problem, max_eval, init_points=rand_evals, log=log)

solvers = {
    'idone': {
        'args': {'--model', '--binarize-categorical', '--rand-evals', '--scaling'},
        'defaults': {
            '--model': 'advanced',
            '--binarize-categorical': 'false',
            '--rand-evals': '1',
            '--scaling': 'false',
            '--internal-logging': 'false'
        },
        'executor': execute_IDONE,
        'check': nop
    },
    'mvrsm': {
        'args': {'--model', '--binarize-categorical', '--rand-evals', '--scaling'},
        'defaults': {
            '--model': 'advanced',
            '--binarize-categorical': 'false',
            '--rand-evals': '1',
            '--scaling': 'true'
        },
        'executor': execute_MVRSM,
        'check': nop
    },
    'sa': {
        'args': set(),
        'defaults': {},
        'executor': execute_SA,
        'check': nop
    },
    'hyperopt': {
        'args': {'--int-conversion-mode', '--rand-evals'},
        'defaults': {
            '--int-conversion-mode': 'quniform',
            '--rand-evals': '3'
        },
        'executor': execute_hyperopt,
        'check': nop
    },
    'randomsearch': {
        'args': {'--int-conversion-mode'},
        'defaults': {
            '--int-conversion-mode': 'quniform'
        },
        'executor': execute_hyperopt_rnd,
        'check': nop
    },
    'pygpgo': {
        'args': {'--rand-evals'},
        'defaults': {
            '--rand-evals': '3',
        },
        'executor': execute_pygpgo,
        'check': nop
    },
    'bayesianoptimization': {
        'args': {'--rand-evals'},
        'defaults': {
            '--rand-evals': '5'
        },
        'executor': execute_bayesianoptimization,
        'check': nop
    },
    'smac': {
        'args': {'--rand-evals', '--deterministic'},
        'defaults': {
            '--rand-evals': '1',
            '--deterministic': 'n'
        },
        'executor': execute_smac,
        'check': check_smac
    },
    'cocabo': {
        'args': {'--rand-evals'},
        'defaults': {
            '--rand-evals': '24'
        },
        'executor': execute_cocabo,
        'check': nop
    }
}

general_args = {'--repetitions', '--max-eval', '--out-path', '--write-every', '--rand-eval-all'}

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
    # Predefined: Synthetic
    print(", ".join(fn.name for fn in fns))
    print(f" (for all: no arguments)")
    print()
    # Predefined: Synthetic
    print(", ".join(fn.name.lower() for fn in dockersimbenches))
    print(f" (for all: no arguments)")
    print(f" (Note: Running these requires docker to be installed.)")
    print()

    # Predefined: CFD Test Problems
    print(f"Solvers:")
    # IDONE
    print(f" idone")
    print(f" --model=<basic|advanced> \t The kind of model IDONE should utilize (default: advanced)")
    print(f" --binarize-categorical=<t|true|f|false> \t Whether to binarize categorical variables. (default: false)")
    print(f" --scaling=<t|true|f|false> \t Whether scaling is applied. (default: false)")
    print(f" --rand-evals=<int> \t Number of random evaluations. (default: 1)")
    print()
    # MVRSM
    print(f" mvrsm")
    print(f" --model=<basic|advanced> \t The kind of model MVRSM should utilize (default: advanced)")
    print(f" --binarize-categorical=<t|true|f|false> \t Whether to binarize categorical variables. (default: false)")
    print(f" --scaling=<t|true|f|false> \t Whether scaling is applied. (default: true)")
    print(f" --rand-evals=<int> \t Number of random evaluations. (default: 1)")
    print()
    # HyperOpt
    print(f" hyperopt")
    print(f" --int-conversion-mode=<quniform|randint> \t The default conversion mode for integers for hyperopt (default: quniform)")
    print(f" --rand-evals=<int> \t Number of random evaluations. (default: 3)")

    print()
    # HyperOpt / randomsearch
    print(f" randomsearch")
    print(f" (no arguments implemented yet)")
    print()
    # pyGPGO
    print(f" pygpgo")
    print(f" --rand-evals=<int> \t Number of random evaluations. (default: 3)")
    print()
    # bayesianoptimization
    print(f" bayesianoptimization")
    print(f" --rand-evals=<int> \t Number of random evaluations. (default: 5)")
    print()
    # smac
    print(f" smac")
    print(f" --rand-evals=<int> \t Number of random evaluations. (default: 1)")
    print()
    # CoCaBO
    print(f" cocabo")
    print(f" --rand-evals=<int> \t Number of random evaluations. (default: 24)")
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
rand_evals_default = general.get('--rand-eval-all')

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

    if rand_evals_default is not None:
        solver['params']['--rand-evals'] = rand_evals_default

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