## This file is a wrapper for the benchmark tool
## to be able to run CoCaBO on these problems.
import tempfile

import numpy as np

from .vendor.methods.CoCaBO import CoCaBO

from collections import Counter
from ..utils import Monitor

def get_variable_type(problem, varidx):
    # Vartype can be 'cont' or 'int'
    vartype = problem.vartype()[varidx]
    
    if vartype == 'cont':
        return 'continuous'
    elif vartype == 'int':
        # No integer support?
        return 'categorical'
    else:
        raise ValueError(f'Variable of type {vartype} supported by CoCaBO.')

def get_variable_domain(problem, varidx):
    # Vartype can be 'cont' or 'int'
    vartype = problem.vartype()[varidx]

    lbs = problem.lbs()
    ubs = problem.ubs()

    if vartype == 'cont':
        return (lbs[varidx], ubs[varidx])
    elif vartype == 'int':
        return tuple(i for i in range(lbs[varidx].astype(int), ubs[varidx].astype(int) + 1))
    else:
        raise ValueError(f'Variable of type {vartype} supported by CoCaBO.')


def get_variables(problem):
    return [
        {
            'name': f'v{i}',
            'type': get_variable_type(problem, i), 
            'domain': get_variable_domain(problem, i)
        } for i in range(problem.dims())
    ]

def optimize_CoCaBO(problem, max_evals, init_points=24, log=None):
    d = problem.dims()

    variables = get_variables(problem)

    # The number of categories for each categorical variable.
    # Why doesn't CoCaBO simply do this:
    C = [len(var['domain']) for var in variables if var['type'] == 'categorical']
    Cmap = [var['domain'] for var in variables if var['type'] == 'categorical']
    # I do not know.
    assert len(C) < d, "CoCaBO requires at least one variable to be continuous."
    # assert len(C) > 0, "CoCaBO on continuous variables only..."
    max_eval_budget = max_evals - init_points
    assert max_eval_budget > 0

    # Compute the permutation and its inverse
    # CoCaBO reorders such that categorical (int) comes first
    # and continuous comes second.
    # This computes the inverse permutation such we can reverse this
    # operation.
    # Note: If the variables end up being mixed up, this is the likely
    #       cause.
    perm = list(range(d))
    vartypes = problem.vartype()
    perm.sort(key=lambda i: 0 if vartypes[i] == "int" else 1)
    perm = np.asarray(perm)
    invperm = np.zeros(d, dtype=int)
    for i, p in enumerate(perm):
        invperm[p] = i

    mon = Monitor(f"CoCaBO", problem, log=log)
    def f(cat, cont):
        # CoCaBO reorders the input so that categorical comes first
        # continuous second.
        # We need to reconstruct the vector...
        xvec = np.concatenate([[Cmap[i][cv] if Cv else Cmap[i][0] for i, cv in enumerate(cat)], cont])[invperm]
        print(xvec)
        mon.commit_start_eval()
        r = problem.evaluate(xvec)
        mon.commit_end_eval(xvec, r)
        return r

    tempdir = tempfile.mkdtemp()
    mon.start()
    optim = CoCaBO(f, init_points, variables, 'LCB', C)
    # Set saving path to a temporary directory.
    # Normally this is set by run trails...
    optim.saving_path = tempdir
    # Set this to ensure it does not crash in the last step.
    # Normally this is set by run trails...
    optim.trial_num = 1
    seed = 0
    try:
        df = optim.runOptim(max_eval_budget, seed)
        mon.end()

        _lbtch, _trls, _li, solY, solX = optim.best_val_list[-1]
    except e:
        mon.end()
        print("An error has occurred while using CoCaBO, terminating early...")
        return None, None, mon
        
    return solX, solY, mon