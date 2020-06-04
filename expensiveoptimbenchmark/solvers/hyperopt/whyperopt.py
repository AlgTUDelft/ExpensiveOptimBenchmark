import numpy as np

from hyperopt import fmin, tpe, rand, hp, STATUS_OK, Trials
from functools import partial

from ..utils import Monitor

def get_variable(problem, varidx):
    vartype = problem.vartype()[varidx]
    lb = problem.lbs()[varidx]
    ub = problem.ubs()[varidx]

    if vartype == 'cont':
        return hp.uniform(f'v{varidx}', lb, ub) 
    elif vartype == 'int':
        # No integer support?
        return hp.quniform(f'v{varidx}', lb, ub, 1) 
    else:
        raise ValueError(f'Variable of type {vartype} supported by HyperOpt (or not added to the converter yet).')


def get_variables(problem):
    return [
        get_variable(problem, i) for i in range(problem.dims())
    ]

def optimize_hyperopt_tpe(problem, max_evals, random_init_evals = 3, log=None):
    variables = get_variables(problem)

    mon = Monitor("hyperopt/tpe", problem, log=log)
    def f(x):
        mon.commit_start_eval()
        r = problem.evaluate(x)
        mon.commit_end_eval(x, r)
        return {
            'loss': r,
            'status': STATUS_OK
        }

    ho_algo = partial(tpe.suggest, n_startup_jobs=rand_evals)
    trials = Trials()

    mon.start()
    ho_result = fmin(f, variables, ho_algo, max_evals=max_evals - random_init_evals, trials=trials)
    mon.end()

    best_trial = trials.best_trial
    # print(f"Best trial: {best_trial}")

    solX = [v for (k, v) in ho_result.items()] 
    # print(f"Best point: {solX}")
    solY = best_trial['result']['loss']

    return solX, solY, mon