import numpy as np

from hyperopt import fmin, tpe, rand, hp, STATUS_OK, Trials
from functools import partial

from ..utils import Monitor

def get_variable(problem, varidx, params):
    vartype = problem.vartype()[varidx]
    lb = problem.lbs()[varidx]
    ub = problem.ubs()[varidx]

    if vartype == 'cont':
        return hp.uniform(f'v{varidx}', lb, ub) 
    elif vartype == 'int':
        icm = params.get('int_conversion_mode')
        if icm == 'randint':
            return hp.randint(f'v{varidx}', int(ub - lb + 1))
        elif icm == 'quniform':
            return hp.quniform(f'v{varidx}', int(lb), int(ub), 1)
        else:
            raise ValueError(f'Unknown int conversion rule. Try using `quniform` or `randint`.')
    elif vartype == 'cat':
        return hp.randint(f'v{varidx}', int(ub - lb + 1))
    else:
        raise ValueError(f'Variable of type {vartype} supported by HyperOpt (or not added to the converter yet).')


def get_variables(problem, params={}):
    return [
        get_variable(problem, i, params) for i in range(problem.dims())
    ]

def optimize_hyperopt_tpe(problem, max_evals, random_init_evals = 3, cparams={}, log=None):
    variables = get_variables(problem, cparams)

    shift = np.zeros(problem.dims())
    if cparams.get('int_conversion_mode') == 'randint':
        idxs = problem.vartype() == 'int'
        shift[idxs] = problem.lbs()[idxs]

    idxs = problem.vartype() == 'cat'
    shift[idxs] = problem.lbs()[idxs]

    mon = Monitor("hyperopt/tpe", problem, log=log)
    def f(x):
        xalt = x + shift
        mon.commit_start_eval()
        r = problem.evaluate(xalt)
        mon.commit_end_eval(xalt, r)
        return {
            'loss': r,
            'status': STATUS_OK
        }

    ho_algo = partial(tpe.suggest, n_startup_jobs=random_init_evals)
    trials = Trials()

    mon.start()
    ho_result = fmin(f, variables, ho_algo, max_evals=max_evals, trials=trials)
    mon.end()

    best_trial = trials.best_trial
    # print(f"Best trial: {best_trial}")

    solX = [v for (k, v) in ho_result.items()] 
    # print(f"Best point: {solX}")
    solY = best_trial['result']['loss']

    return solX, solY, mon

def optimize_hyperopt_rnd(problem, max_evals, cparams={}, log=None):
    variables = get_variables(problem, cparams)

    shift = np.zeros(problem.dims())
    if cparams.get('int_conversion_mode') == 'randint':
        idxs = problem.vartype() == 'int'
        shift[idxs] = problem.lbs()[idxs]

    idxs = problem.vartype() == 'cat'
    shift[idxs] = problem.lbs()[idxs]

    mon = Monitor("hyperopt/randomsearch", problem, log=log)
    def f(x):
        xalt = x + shift
        mon.commit_start_eval()
        r = problem.evaluate(xalt)
        mon.commit_end_eval(xalt, r)
        return {
            'loss': r,
            'status': STATUS_OK
        }

    ho_algo = rand.suggest
    trials = Trials()

    mon.start()
    ho_result = fmin(f, variables, ho_algo, max_evals=max_evals, trials=trials)
    mon.end()

    best_trial = trials.best_trial
    # print(f"Best trial: {best_trial}")

    solX = [v for (k, v) in ho_result.items()] 
    # print(f"Best point: {solX}")
    solY = best_trial['result']['loss']

    return solX, solY, mon