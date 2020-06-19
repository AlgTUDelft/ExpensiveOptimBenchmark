import numpy as np

from .MVRSM import MVRSM_minimize
from ..utils import Monitor, Binarizer
from collections import Counter

def optimize_MVRSM(problem, max_evals, model='advanced', binarize_categorical=False, log=None):
    d = problem.dims()

    vartypes = problem.vartype()

    # Find the permutation that lists integers first.
    # And its inverse.
    perm = list(range(d))
    perm.sort(key=lambda i: 0 if vartypes[i] == "int" or vartypes[i] == "cat" else 1)
    perm = np.asarray(perm)
    invperm = np.zeros(d, dtype=int)
    for i, p in enumerate(perm):
        invperm[p] = i

    counted = Counter(vartypes)
    num_int = counted["int"] + counted["cat"]
    assert num_int == d

    # Bounds are reordered
    lb = problem.lbs()[perm]
    ub = problem.ubs()[perm]

    # Generate initial point, round the integers.
    x0 = np.random.rand(d)*(ub-lb) + lb
    x0[0:num_int] = np.round(x0[0:num_int])

    if binarize_categorical:
        b = Binarizer(problem.vartype()[perm] == 'cat', lb, ub)
        x0 = b.binarize(x0)
        # Get new bounds.
        lb = b.lbs()
        ub = b.ubs()
        # All new variables introduced by the binarized are 'integer'.
        num_int += b.dout - b.din
        # print(f"Binarization introduced {b.dout - b.din} new variables.")

    mon = Monitor(f"MVRSM/{model}{'/binarized' if binarize_categorical else ''}", problem, log=log)
    def f(x):
        if binarize_categorical:
            xred = b.unbinarize(x)[invperm]
        else:
            xred = x[invperm]
        # print(xred)
        mon.commit_start_eval()
        r = problem.evaluate(xred)
        mon.commit_end_eval(xred, r)
        return r
    
    mon.start()
    solX, solY, model, logfile = MVRSM_minimize(f, x0, lb, ub, num_int, max_evals, model_type=model)
    mon.end()

    return solX[invperm], solY, mon