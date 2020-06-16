import numpy as np

from pyGPGO.GPGO import GPGO
from ..utils import Monitor

def get_variable_type(problem, varidx):
    # Vartype can be 'cont' or 'int'
    vartype = problem.vartype()[varidx]
    
    if vartype == 'cont':
        return 'cont'
    elif vartype == 'int' or vartype == 'cat':
        # Note: integer support is wonky with pyGPGO.
        # Additional steps (eg. rounding) in objective function may be required.
        return 'cont'
    else:
        raise ValueError(f'Variable of type {vartype} supported by pyGPGO.')

def get_variable_domain(problem, varidx):
    # Vartype can be 'cont' or 'int'
    vartype = problem.vartype()[varidx]

    lbs = problem.lbs()
    ubs = problem.ubs()

    if vartype == 'cont' or vartype == 'int':
        return [lbs[varidx], ubs[varidx]]
    else:
        raise ValueError(f'Variable of type {vartype} supported by pyGPGO.')

def get_variables(problem):
    return {
        f'v{i}': (get_variable_type(problem, i), get_variable_domain(problem, i))
        for i in range(problem.dims())
    }

# Note: one has to specify a Gaussian Process and Acquisition function
# for pyGPGO.
def optimize_pyGPGO(problem, max_evals, gp, acq, random_init_evals = 3, log=None):
    params = get_variables(problem)
    
    mon = Monitor("pyGPGO/GP/matern/EI", problem, log=log)

    # Note, pyGPGO seems to maximize by default, objective is therefore negated.
    # Furthermore: passing `int` as type seems to be very fragile.
    # performing manual rounding instead.
    def f(**x):
        mon.commit_start_eval()
        xvec = np.array([v if t == 'cont' else round(v) for (k, v), t in zip(x.items(), problem.vartype())])
        # print(f"Processed vector: {xvec}")
        r = problem.evaluate(xvec)
        mon.commit_end_eval(xvec, r)
        return -float(r)

    mon.start()
    gpgo = GPGO(gp, acq, f, params)
    gpgo.run(max_iter = max_evals - random_init_evals, init_evals=random_init_evals)
    mon.end()
    solX, solY = gpgo.getResult()

    return solX, -solY, mon