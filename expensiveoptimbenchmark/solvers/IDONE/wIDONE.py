import numpy as np

from .IDONE import IDONE_minimize
from ..utils import Monitor

def optimize_IDONE(problem, max_evals, model='advanced', log=None):
    d = problem.dims()
    lb = problem.lbs()
    ub = problem.ubs()

    if not all(np.logical_or(problem.vartype() == 'int', problem.vartype() == 'cat')):
        raise ValueError(f'Variable of type {vartype} supported by IDONE.')
    
    x0 = np.round(np.random.rand(d)*(ub-lb) + lb)

    mon = Monitor(f"IDONE/{model}", problem, log=log)
    def f(x):
        mon.commit_start_eval()
        r = problem.evaluate(x)
        mon.commit_end_eval(x, r)
        return r
    
    mon.start()
    solX, solY, model, logfile = IDONE_minimize(f, x0, lb, ub, max_evals, model_type=model)
    mon.end()

    return solX, solY, mon