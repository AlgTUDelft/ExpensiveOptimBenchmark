import numpy as np
from tqdm import tqdm

from julia import Main
Main.include("./expensiveoptimbenchmark/solvers/DONEjl/vendor/DONEs.jl")
DONEs = Main.DONEs

def minimize_DONEjl(f, lb, ub, rand_evals, max_evals, hyperparams, progressbar=True):
    n_vars      = len(lb)
    n_basis     = hyperparams.get('n_basis', 1000) # larger with more n_vars (high dim)
    sigma_coeff = hyperparams.get('sigma_coeff', 0.1) # / sqrt(n_vars)
    # 0.1 for n_vars < 10
    # after that, scale with square root.
    sigma_def   = np.minimum(0.1, 0.1 / np.sqrt(n_vars) * np.sqrt(10))
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

    supported = (problem.vartype() == 'cont')

    if not (supported).all():
        raise ValueError(f'Variable of types {np.unique(problem.vartype()[np.logical_not(supported)])} are not supported by DONEjl.')

    mon = Monitor(f"DONEjl", problem, log=log)
    def f(x):
        mon.commit_start_eval()
        r = problem.evaluate(x)
        mon.commit_end_eval(x, r)
        return r
    
    mon.start()
    solX, solY, model = minimize_DONEjl(f, lb, ub, rand_evals, max_evals, hyperparams)
    mon.end()

    return solX, solY, mon