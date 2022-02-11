import numpy as np

from cma import BoundPenalty, BoundTransform, fmin, CMAOptions
from ..utils import Monitor, Binarizer

def optimize_CMA(problem, max_evals, bound_h='transform', binarize_categorical=False, log=None):
    d = problem.dims()

    vartypes = problem.vartype()

    if binarize_categorical:
        raise Exception("While sort-of implemented, this implementation has issues. Please do not enable binarization.")

    # Bounds
    lb = problem.lbs()
    ub = problem.ubs()

    # Generate initial point, round the integers.

    if binarize_categorical:
        b = Binarizer(problem.vartype()[perm] == 'cat', lb, ub)
        x0 = b.binarize(x0)
        # Get new bounds.
        lb = b.lbs()
        ub = b.ubs()
        # All new variables introduced by the binarized are 'integer'.
        # print(f"Binarization introduced {b.dout - b.din} new variables.")
        vartypes = vartypes[self.origin_mapping]
    
    cmascale = (ub - lb)
    x0 = (0.5*(ub-lb) + lb) / cmascale
    not_continuous_variables = vartypes != "cont"
    x0[not_continuous_variables] = np.round(x0[not_continuous_variables])
    sigma0 = 1/4

    mon = Monitor(f"CMAES{'/cbinarized' if binarize_categorical else ''}/{bound_h}", problem, log=log)
    def f(x):
        x = x * cmascale
        if binarize_categorical:
            xred = b.unbinarize(x)
        else:
            xred = x
        x = x[vartypes != "cont"]
        # print(xred)
        mon.commit_start_eval()
        r = problem.evaluate(xred)
        mon.commit_end_eval(xred, r)
        return r
    
    # lb, ub, max_evals
    opts = CMAOptions()
    opts['bounds'] = [list(lb / cmascale), list(ub / cmascale)]
    opts['BoundaryHandler'] = BoundPenalty if bound_h == "penalty" else BoundTransform
    opts['maxfevals'] = max_evals
    opts['verbose'] = 0 # Supress printing!
    opts['integer_variables'] = [i for i,t in enumerate(vartypes) if t != 'cont']
    mon.start()
    res = fmin(f, x0 / cmascale, sigma0, options=opts)
    mon.end()
    
    # """
    # - `res[0]` (`xopt`) -- best evaluated solution  
    # - `res[1]` (`fopt`) -- respective function value  
    # - `res[2]` (`evalsopt`) -- respective number of function evaluations  
    # - `res[3]` (`evals`) -- number of overall conducted objective function evaluations  
    # - `res[4]` (`iterations`) -- number of overall conducted iterations  
    # - `res[5]` (`xmean`) -- mean of the final sample distribution  
    # - `res[6]` (`stds`) -- effective stds of the final sample distribution  
    # - `res[-3]` (`stop`) -- termination condition(s) in a dictionary  
    # - `res[-2]` (`cmaes`) -- class `CMAEvolutionStrategy` instance  
    # - `res[-1]` (`logger`) -- class `CMADataLogger` instance
    # """

    return mon.best_x, mon.best_fitness, mon