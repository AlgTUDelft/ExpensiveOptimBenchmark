import numpy as np
import math
from ..utils import Monitor
from scipy.optimize import basinhopping, minimize, Bounds

def get_variable_bounds(problem):
    lbs = problem.lbs()
    ubs = problem.ubs()

    return Bounds(lbs, ubs)

def optimize_basinhopping(problem, max_evals, T=1.0, stepsize=0.5, localmethod="L-BFGS-B", log=None, verbose=True):
    vt = problem.vartype()
    lbs = problem.lbs()
    ubs = problem.ubs()
    mon = Monitor(f"scipy.basinhopping/{localmethod}", problem, log=log)
    def f(x):
        # scipy.optimize
        xvec = x.copy()
        # Round non-continuous variables
        xvec[vt != 'cont'] = np.round(xvec[vt != 'cont'])
        # Clamp variable values to bounds
        np.clip(xvec, lbs, ubs, out=xvec)
        mon.commit_start_eval()
        r = problem.evaluate(xvec)
        mon.commit_end_eval(xvec, r)
        return r
    
    def budget_check_global(x, f, accept):
        # Callback used to stop basin hopping when evaluation limit is reached.
        # x -- local minimum solution
        # f -- corresponding fitness
        # accept -- whether this local optima was accepted as the new reference solution
        return mon.num_iters >= max_evals

    def budget_check_local(x):
        # Callback used to stop local optimization when evaluation limit is reached.
        # x -- local minimum solution
        return mon.num_iters >= max_evals
    
    minimizer_kwargs = {
        'method': localmethod,
        'bounds': get_variable_bounds(problem),
        'callback': budget_check_local
    }

    # Generate initial point
    lb = problem.lbs()
    ub = problem.ubs()
    d = len(lb)
    x0 = np.random.rand(d)*(ub-lb) + lb
    x0[vt != 'cont'] = np.round(x0[vt != 'cont'])

    mon.start()
    optim_result = basinhopping(func=f, x0=x0, niter=max_evals, T=T, stepsize=stepsize, minimizer_kwargs=minimizer_kwargs, callback=budget_check_global)
    mon.end()

    solX = optim_result['x']
    solY = optim_result['fun']

    return solX, solY, mon


def optimize_scipy_local(problem, max_evals, method="BFGS", log=None, verbose=False):

    vt = problem.vartype()
    mon = Monitor(f"scipy.{method}", problem, log=log)
    def f(x):
        # scipy.optimize
        xvec = x.copy()
        # Round non-continuous variables
        xvec[vt != 'cont'] = np.round(xvec[vt != 'cont'])
        mon.commit_start_eval()
        r = problem.evaluate(xvec)
        mon.commit_end_eval(xvec, r)
        return r
    
    def budget_check_local(x):
        # Callback used to stop local optimization when evaluation limit is reached.
        # x -- local minimum solution
        return mon.num_iters >= max_evals

    # Generate initial point, round the integers.
    lb = problem.lbs()
    ub = problem.ubs()
    x0 = np.random.rand(d)*(ub-lb) + lb
    x0[0:num_int] = np.round(x0[0:num_int])

    mon.start()
    optim_result = minimize(func=f, x0=x0, method=method, bounds=get_variable_bounds(problem), callback=budget_check_local)
    mon.end()

    solX = optim_result['x']
    solY = optim_result['fun']

    return solX, solY, mon