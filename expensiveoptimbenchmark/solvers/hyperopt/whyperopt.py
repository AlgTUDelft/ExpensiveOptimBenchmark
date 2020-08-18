import numpy as np

from hyperopt import fmin, tpe, rand, hp, STATUS_OK, Trials
from functools import partial

from ..utils import Monitor

def get_variable_simple(problem, varidx, params):
    vartype = problem.vartype()[varidx]
    lb = problem.lbs()[varidx]
    ub = problem.ubs()[varidx]

    if vartype == 'cont':
        return hp.uniform(f'v{varidx}', lb, ub) 
    elif vartype == 'int':
        icm = params.get('int_conversion_mode')
        if icm == 'randint':
            return hp.randint(f'v{varidx}', int(ub - lb + 1))
        elif icm == 'quniform':
            return hp.quniform(f'v{varidx}', int(lb), int(ub), 1)
        else:
            raise ValueError(f'Unknown int conversion rule. Try using `quniform` or `randint`.')
    elif vartype == 'cat':
        return hp.randint(f'v{varidx}', int(ub - lb + 1))
    else:
        raise ValueError(f'Variable of type {vartype} supported by HyperOpt (or not added to the converter yet).')

def get_variable_with_dependencies(problem, varidx, dependencies, allvars, params):
    vartype = problem.vartype()[varidx]
    assert vartype in ['int', 'cat']
    lb = problem.lbs()[varidx]
    ub = problem.ubs()[varidx]

    choices = [
        {varidx: i}
    for i in range(int(lb), int(ub)+1)]

    # Add dependencies to choices
    for (dependency, values) in dependencies[0].items():
        var = allvars[dependency]
        for value in values:
            choices[int(value)][dependency] = var

    return hp.choice(f'c{varidx}', choices)

def get_variables(problem, params={}):
    """
    Create a hyperopt search space description for problem.

    Parameters
    ----
    params: dict - specifies options to manipulate how kinds of variables are encoded.
    > int_conversion_mode - 
    > The variable type `int` is by default encoded using `quniform`, setting
    > this variable to `randint` will make use of randint instead.

    A few notes on how this uses hyperopt's variable encoding mechanism to encode
    a tree of dependent variables. 
    In order for a variable to be dependent, it should only occur within an
    `hp.choice`, as such we cannot use a vector (as we did previously).
    
    Instead we make use of dictionaries, where the index of the variable is the
    key and the variable is the value. `hp.choice` gets to pick between options
    that utilize the same rule, allowing us to 'flatten' the dict of values and
    dicts that hyperopt will pass our evaluation function.

    Eg. if we have two binary variables 0 and 1, 1 being conditional on 0 being 1.
    The resulting hyperopt specification should be equivalent to
    ```
    {0: hp.choice([{0: 0}, {0: 1, 1: hp.randint(1+1)}])}
    ```
    Resulting in hyperopt passing a value like this:
    ```
    {0: {0: 1, 1: 0}} or {0: {0: 0}}
    ```
    Flattening would result in
    ```
    {0: 1, 1: 0} or {0: 0}
    ```
    Which can allow us to recover a vector (potentially containing None values)
    ```
    [1, 0] or [0, None]
    ```
    """
    dependencies = problem.dependencies()
    vartypes = problem.vartype()

    # The roots of the dependency tree.
    roots = []
    dependents = {}
    for i, dependency in enumerate(dependencies):
        if dependency is None:
            # This variable is not dependent on anything.
            roots.append(i)
            continue
        on = dependency['on']
        values = dependency['values']
        # A variable being depended on, need to be categorical (or maybe integer)
        assert vartypes[on] in ['int', 'cat']
        deps, remaining = dependents.get(on, ({}, 0))
        deps[i] = values
        dependents[on] = (deps, remaining + 1)
    #
    available = [i for i in range(len(dependencies)) if i not in dependents]
    allvars = [None for _ in range(len(dependencies))]

    while len(available) > 0:
        i = available.pop()
        df = dependents.get(i)
        if df is None:
            allvars[i] = get_variable_simple(problem, i, params)
            # assert allvars[i] is not None
        else:
            allvars[i] = get_variable_with_dependencies(problem, i, df, allvars, params)
            # assert allvars[i] is not None
        # Update remaining number of required variables (if any)
        mydeps = dependencies[i]
        if mydeps is not None:
            on = mydeps['on']
            deps, remaining = dependents[on]
            remaining = remaining - 1
            dependents[on] = (deps, remaining)
            # print(f"Resolved a dependency for {on}, {remaining} remaining.")
            if remaining == 0:
                available.append(on)
    
    # assert all(allvars[i] is not None for i in range(len(dependencies)))

    return { num: allvars[num] for num in roots }

def dict_of_dicts_to_vec(d: int, dd: dict):
    flatdict = {}
    tovisit = [dd]
    
    while len(tovisit) > 0:
        dct = tovisit.pop()
        flatdict.update(dct)
        for (key, value) in dct.items():
            if type(value) is dict:
                tovisit.append(value)
    
    result = [None for _ in range(d)]
    for (key, value) in flatdict.items():
        result[key] = value
    
    return np.asarray(result)

def optimize_hyperopt_tpe(problem, max_evals, random_init_evals = 3, cparams={}, log=None):
    variables = get_variables(problem, cparams)
    d = problem.dims()
    
    shift = np.zeros(d)
    if cparams.get('int_conversion_mode') == 'randint':
        idxs = problem.vartype() == 'int'
        shift[idxs] = problem.lbs()[idxs]

    idxs = problem.vartype() == 'cat'
    shift[idxs] = problem.lbs()[idxs]

    mon = Monitor("hyperopt/tpe", problem, log=log)
    def f(x):
        xalt = dict_of_dicts_to_vec(d, x)
        xaltnotnone = xalt != None
        xalt[xaltnotnone] = xalt[xaltnotnone] + shift[xaltnotnone]

        mon.commit_start_eval()
        r = problem.evaluate(xalt)
        mon.commit_end_eval(xalt, r)
        return {
            'loss': r,
            'status': STATUS_OK
        }

    ho_algo = partial(tpe.suggest, n_startup_jobs=random_init_evals)
    trials = Trials()

    mon.start()
    ho_result = fmin(f, variables, ho_algo, max_evals=max_evals, trials=trials)
    mon.end()

    best_trial = trials.best_trial
    # print(f"Best trial: {best_trial}")

    solX = [v for (k, v) in ho_result.items()] 
    # print(f"Best point: {solX}")
    solY = best_trial['result']['loss']

    return solX, solY, mon

def optimize_hyperopt_rnd(problem, max_evals, cparams={}, log=None):
    variables = get_variables(problem, cparams)
    d = problem.dims()
    
    shift = np.zeros(d)
    if cparams.get('int_conversion_mode') == 'randint':
        idxs = problem.vartype() == 'int'
        shift[idxs] = problem.lbs()[idxs]

    idxs = problem.vartype() == 'cat'
    shift[idxs] = problem.lbs()[idxs]

    mon = Monitor("hyperopt/randomsearch", problem, log=log)
    def f(x):
        xalt = dict_of_dicts_to_vec(d, x)
        xaltnotnone = xalt != None
        xalt[xaltnotnone] = xalt[xaltnotnone] + shift[xaltnotnone]

        mon.commit_start_eval()
        r = problem.evaluate(xalt)
        mon.commit_end_eval(xalt, r)
        return {
            'loss': r,
            'status': STATUS_OK
        }

    ho_algo = rand.suggest
    trials = Trials()

    mon.start()
    ho_result = fmin(f, variables, ho_algo, max_evals=max_evals, trials=trials)
    mon.end()

    best_trial = trials.best_trial
    # print(f"Best trial: {best_trial}")

    solX = [v for (k, v) in ho_result.items()] 
    # print(f"Best point: {solX}")
    solY = best_trial['result']['loss']

    return solX, solY, mon