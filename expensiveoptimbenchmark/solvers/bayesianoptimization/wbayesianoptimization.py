import numpy as np
import math
from ..utils import Monitor
from bayes_opt import BayesianOptimization

def get_variable_domain(problem, varidx):
    # Vartype can be 'cont' or 'int'
    vartype = problem.vartype()[varidx]

    lbs = problem.lbs()
    ubs = problem.ubs()
    
    return (lbs[varidx], ubs[varidx] + (1 if vartype != 'cont' else 0))

def get_variables(problem):
    n = problem.dims()
    nlog10 = math.ceil(math.log10(n))

    return {
        f'v{i:0{nlog10}}': get_variable_domain(problem, i)
        for i in range(problem.dims())
    }

def optimize_bayesian_optimization(problem, max_evals, random_init_evals = 5, log=None):
    variables = get_variables(problem)
    n = problem.dims()

    mon = Monitor("bayesianoptimization", problem, log=log)
    def f(**x):
        # As with pyGPGO, bayesianoptimisation does not naturally support integer variables.
        # As such we round them.
        xvec = np.array([v for (k, v), t in zip(x.items(), problem.vartype())])
        mon.commit_start_eval()
        r = problem.evaluate(xvec)
        mon.commit_end_eval(xvec, r)
        # Negate because bayesianoptimization maximizes by default.
        # And optimizer.minimize does not actually exist.
        # Include some random noise to avoid issues if all samples are the same.
        eps = 1e-4
        return -r + np.random.standard_normal() * eps
    mon.start()
    nlog10 = math.ceil(math.log10(n))
    optimizer = BayesianOptimization(
        f=f,
        pbounds=get_variables(problem),
        ptypes={f'v{i:0{nlog10}}': float if problem.vartype()[i] == 'cont' else int for i in range(problem.dims())}
    )

    optimizer.maximize(
        init_points=random_init_evals,
        n_iter=max_evals - random_init_evals)
    mon.end()

    solX = [v for (k, v) in optimizer.max['params'].items()] 
    solY = optimizer.max['target']

    return solX, solY, mon