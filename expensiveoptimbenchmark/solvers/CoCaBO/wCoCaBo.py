## Note: No wrapper for CoCaBO has been written yet.
## Only this variable conversion mechanism.

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

    if vartype == 'cont' or vartype == 'int':
        return tuple(lbs[varidx], ubs[varidx])
    elif :
        return tuple(i for i in range(lbs[varidx], ubs[varidx] + 1))
    else:
        raise ValueError(f'Variable of type {vartype} supported by CoCaBO.')


def get_variables(problem):
    return [
        { 'name': f'v{i}', 'type': '' } for i in range(problem.dims())
    ]