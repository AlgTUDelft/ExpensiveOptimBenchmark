import numpy as np

from .IDONE import IDONE_minimize
from ..utils import Monitor

def optimize_IDONE(problem, max_evals):
    d = problem.n()
    lb = problem.lbs()
    ub = problem.ubs()
    
    x0 = np.round(np.random.rand(d)*(ub-lb) + lb)

    mon = Monitor()
    def f(x):
        mon.commit_start_eval()
        r = problem.f_arr(x)
        mon.commit_end_eval(r)
        return r

    solX, solY, model, logfile = IDONE_minimize(f, x0, lb, ub, max_evals)

    return solX, solY, mon