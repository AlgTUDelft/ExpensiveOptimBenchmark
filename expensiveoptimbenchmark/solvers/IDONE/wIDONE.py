import numpy as np

from .IDONE import IDONE_minimize
from ..utils import Monitor, Binarizer

def optimize_IDONE(problem, max_evals, rand_evals=0, model='advanced', binarize_categorical=False, binarize_int=False, thompson_sampling = False, enable_scaling=False, log=None):
    d = problem.dims()
    lb = problem.lbs()
    ub = problem.ubs()

    supported = (np.logical_or(problem.vartype() == 'int', problem.vartype() == 'cat'))

    if not (supported).all():
        raise ValueError(f'Variable of types {np.unique(problem.vartype()[np.logical_not(supported)])} are not supported by IDONE.')
    
    x0 = np.round(np.random.rand(d)*(ub-lb) + lb)

    # Don't enable binarization if there is nothing to binarize.
    binarize_categorical = binarize_categorical and (problem.vartype() == 'cat').any()
    binarize_int = binarize_int and (problem.vartype() == 'int').any()

    if binarize_categorical or binarize_int:
        if binarize_categorical and binarize_int:
            b = Binarizer(problem.vartype() == 'cat' or problem.vartype() == 'int', lb, ub)
        elif binarize_categorical:
            b = Binarizer(problem.vartype() == 'cat', lb, ub)
        else:
            b = Binarizer(problem.vartype() == 'int', lb, ub)
        x0 = b.binarize(x0)
        # Get new bounds.
        lb = b.lbs()
        ub = b.ubs()


    mon = Monitor(f"IDONE/{model}{'/scaled' if enable_scaling else ''}{'/binarized' if binarize_categorical or binarize_int else ''}{'/TS' if thompson_sampling else ''}", problem, log=log)
    def f(x):
        mon.commit_start_eval()
        if binarize_categorical or binarize_int:
            xalt = b.unbinarize(x)
        else:
            xalt = x
        r = problem.evaluate(xalt)
        mon.commit_end_eval(xalt, r)
        return r
    
    mon.start()
    solX, solY, model, logfile = IDONE_minimize(f, x0, lb, ub, max_evals, rand_evals=rand_evals, enable_scaling=enable_scaling, model_type=model, thompson_sampling=thompson_sampling)
    mon.end()

    return solX, solY, mon