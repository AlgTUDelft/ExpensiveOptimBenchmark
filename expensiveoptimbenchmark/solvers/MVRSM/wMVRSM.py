import numpy as np

from .MVRSM import MVRSM_minimize
from ..utils import Monitor
from collections import Counter

def optimize_MVRSM(problem, max_evals, model='advanced', log=None):
    d = problem.dims()

    vartypes = problem.vartype()

    # Find the permutation that lists integers first.
    # And its inverse.
    perm = list(range(d))
    perm.sort(key=lambda i: 0 if vartypes[i] == "int" else 1)
    perm = np.asarray(perm)
    invperm = np.zeros(d, dtype=int)
    for i, p in enumerate(perm):
        invperm[p] = i
    num_int = Counter(vartypes)["int"]

    # Bounds are reordered
    lb = problem.lbs()[perm]
    ub = problem.ubs()[perm]
    
    # Generate initial point, round the integers.
    x0 = np.random.rand(d)*(ub-lb) + lb
    x0[0:num_int] = np.round(x0[0:num_int])

    mon = Monitor(f"MVRSM/{model}", problem, log=log)
    def f(x):
        xred = x[invperm]
        mon.commit_start_eval()
        r = problem.evaluate(xred)
        mon.commit_end_eval(xred, r)
        return r
    
    mon.start()
    solX, solY, model, logfile = MVRSM_minimize(f, x0, lb, ub, num_int, max_evals, model_type=model)
    mon.end()

    return solX[invperm], solY, mon