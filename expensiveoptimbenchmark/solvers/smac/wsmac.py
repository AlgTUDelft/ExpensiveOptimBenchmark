import numpy as np
import math

# Variable types
from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter

# 
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.scenario.scenario import Scenario

from ..utils import Monitor

def get_variable(problem, varidx):
    vartype = problem.vartype()[varidx]
    lb = problem.lbs()[varidx]
    ub = problem.ubs()[varidx]
    n = len(problem.vartype())
    nlog10 = math.ceil(math.log10(n))

    if vartype == 'cont':
        return UniformFloatHyperparameter(f'v{varidx:0{nlog10}}', lb, ub) 
    elif vartype == 'int':
        return UniformIntegerHyperparameter(f'v{varidx:0{nlog10}}', lb, ub) 
    else:
        raise ValueError(f'Variable of type {vartype} supported by SMAC (or not added to the converter yet).')


def get_variables(problem):
    cs = ConfigurationSpace()
    for i in range(problem.dims()):
        cs.add_hyperparameter(get_variable(problem, i))

    return cs

def optimize_smac(problem, max_evals, log=None):

    mon = Monitor("smac", problem, log=log)
    def f(cfg):
        xvec = np.array([cfg[k] for k, t in zip(cfg, problem.vartype())])
        mon.commit_start_eval()
        r = problem.evaluate(xvec)
        mon.commit_end_eval(xvec, r)
        return r

    sc = Scenario({
        "run_obj": "quality",
        "runcount-limit": max_evals,
        "cs": get_variables(problem),
        # "use_pynisher": False,
        "output_dir": None,
        # "deterministic": False # FIXME: Assume noise by default?
    })
    # smac = SMAC4HPO(scenario=sc, tae_runner=f)
    smac = SMAC4HPO(scenario=sc, tae_runner=f)


    mon.start()
    result = smac.optimize()
    mon.end()

    # print(f"Best trial: {best_trial}")

    solX = [v for (k, v) in result.items()] 
    # print(f"Best point: {solX}")
    # Note, this runs the function again, just to compute the fitness again.
    # solY = f(solX)
    # We can also ask it from our evaluation monitor.
    solY = mon.best_fitness

    return solX, solY, mon
