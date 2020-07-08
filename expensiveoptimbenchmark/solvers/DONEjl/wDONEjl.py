import numpy as np
from julia import Main
from tqdm import tqdm

Main.include("./expensiveoptimbenchmark/solvers/DONEjl/vendor/DONEs.jl")
DONEs = Main.DONEs

def minimize_DONEjl(f, lb, ub, max_evals, hyperparams):
    n_vars      = len(lb)
    n_basis     = hyperparams.get('n_basis', 1000) # larger with more n_vars (high dim)
    sigma_coeff = hyperparams.get('sigma_coeff', 0.1) # / sqrt(n_vars)
    # 0.1 for n_vars < 10
    # after that, scale with square root.
    sigma_def   = np.minimum(0.1, 0.1 / np.sqrt(n_vars) * np.sqrt(10))
    sigma_s     = hyperparams.get('sigma_s', sigma_def)
    sigma_f     = hyperparams.get('sigma_f', sigma_def)

    rfe = DONEs.RFE(n_basis, len(lb), sigma_coeff)
    done = DONEs.DONE(rfe, lb.astype(float), ub.astype(float), sigma_s, sigma_f)
    best_x = None
    best_y = np.inf
    for i in tqdm(range(max_evals)):
        xi = DONEs.new_input(done)
        yi = f(xi)
        if yi < best_y:
            best_x = xi
            best_y = yi
        # Note: `!` in julia is replaced by `_b` by pyjulia.
        DONEs.add_measurement_b(done, xi, yi)
        DONEs.update_optimal_input_b(done)
    return best_y, best_x, done

from ..utils import Monitor

def optimize_DONEjl(problem, max_evals, hyperparams, log=None):
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
    solX, solY, model = minimize_DONEjl(f, lb, ub, max_evals, hyperparams)
    mon.end()

    return solX, solY, mon