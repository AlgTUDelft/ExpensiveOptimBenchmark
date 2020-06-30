import numpy as np

from .SA import SA_minimize
from ..utils import Monitor

def optimize_SA(problem, max_evals, log=None):
    d = problem.dims()
    lb = problem.lbs()
    ub = problem.ubs()

    if not (np.logical_or(problem.vartype() == 'int', problem.vartype() == 'cat')).all():
        raise ValueError(f'Variable of type {vartype} supported by SA.')
    
    x0 = np.round(np.random.rand(d)*(ub-lb) + lb)

    mon = Monitor(f"SA", problem, log=log)
    def f(x):
        mon.commit_start_eval()
        r = problem.evaluate(x)
        mon.commit_end_eval(x, r)
        return r
    
    mon.start()
    solY, solX = SA_minimize(f, x0, lb, ub, max_evals)
    mon.end()

    return solX, solY, mon