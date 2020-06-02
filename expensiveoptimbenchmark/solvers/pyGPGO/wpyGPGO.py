import numpy as np

from pyGPGO.GPGO import GPGO
from ..utils import Monitor

def get_variable_type(problem, varidx):
    # Vartype can be 'cont' or 'int'
    vartype = problem.vartype()[varidx]
    
    if vartype == 'cont':
        return 'cont'
    elif vartype == 'int':
        # Note: integer support is wonky with pyGPGO.
        # Additional steps (eg. rounding) in objective function may be required.
        return 'int'
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
def optimize_pyGPGO(problem, max_evals, gp, acq):
    params = get_variables(problem)
    
    mon = Monitor()

    # Note, pyGPGO seems to maximize by default, objective is therefore negated.
    def f(**x):
        mon.commit_start_eval()
        xvec = np.array([v for k, v in x.items()])
        r = -problem.evaluate(xvec)
        mon.commit_end_eval(r)
        return r

    mon.start()
    gpgo = GPGO(gp, acq, f, params)
    gpgo.run(max_iter = max_evals)
    mon.end()
    solX, solY = gpgo.getResult()

    return solX, -solY, mon