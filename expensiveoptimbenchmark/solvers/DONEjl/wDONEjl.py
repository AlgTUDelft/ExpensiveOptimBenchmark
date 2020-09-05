import numpy as np
import math
from tqdm import tqdm
from pathlib import Path

DONEs_loc = Path(__file__).parent.absolute().joinpath("./vendor/DONEs.jl").__str__()

from julia import Main
Main.include(DONEs_loc)
DONEs = Main.DONEs

def minimize_DONEjl(f, lb, ub, rand_evals, max_evals, hyperparams, progressbar=True):
    n_vars      = len(lb)
    n_basis     = hyperparams.get('n_basis', 2000) # generally larger with more n_vars (high dim)
    
    sigma_coeff = hyperparams.get('sigma_coeff', 0.1 if n_vars < 100 else 1 / sqrt(n_vars) )
    
    sigma_def   = 0.1 if n_vars < 100 else 1.0 / math.sqrt(n_vars)
    sigma_s     = hyperparams.get('sigma_s', sigma_def)
    sigma_f     = hyperparams.get('sigma_f', sigma_def)

    rng = np.random.default_rng()

    lbs = np.asarray(lb).astype(float)
    ubs = np.asarray(ub).astype(float)

    rfe = DONEs.RFE(n_basis, len(lb), sigma_coeff)
    done = DONEs.DONE(rfe, lbs, ubs, sigma_s, sigma_f)
    best_x = None
    best_y = np.inf

    if progressbar:
        idxiter = tqdm(range(max_evals))
    else:
        idxiter = range(max_evals)
    
    for i in idxiter:
        if i < rand_evals:
            xi = rng.uniform(lbs, ubs)
            yi = f(xi)
        else:
            xi = DONEs.new_input(done)
            yi = f(xi)
        if yi < best_y:
            best_x = xi
            best_y = yi
        # Note: `!` in julia is replaced by `_b` by pyjulia.
        DONEs.add_measurement_b(done, xi, yi)
        if i >= rand_evals - 1:
            DONEs.update_optimal_input_b(done)
    return best_y, best_x, done

from ..utils import Monitor

def optimize_DONEjl(problem, rand_evals, max_evals, hyperparams, log=None):
    d = problem.dims()
    lb = problem.lbs()
    ub = problem.ubs()

    rounded = (problem.vartype() != 'cont')
    
    mon = Monitor(f"DONEjl", problem, log=log)
    def f(x):
        xr = np.asarray(x)
        xr[rounded] = np.round(xr[rounded])
        mon.commit_start_eval()
        r = problem.evaluate(xr)
        mon.commit_end_eval(xr, r)
        return r
    
    mon.start()
    solX, solY, model = minimize_DONEjl(f, lb, ub, rand_evals, max_evals, hyperparams)
    mon.end()

    return solX, solY, mon