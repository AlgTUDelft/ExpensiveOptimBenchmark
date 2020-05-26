from pyGPGO.GPGO import GPGO
from ..utils import Monitor

# Note: one has to specify a Gaussian Process and Acquisition function
# for pyGPGO.
def optimize_pyGPGO(problem, max_evals, gp, acq):
    params = problem.vars()
    
    mon = Monitor()

    # Note, pyGPGO seems to maximize by default, objective is therefore negated.
    def f(**x):
        mon.commit_start_eval()
        r = -problem.f_kw(**x)
        mon.commit_end_eval(r)
        return r

    mon.start()
    gpgo = GPGO(gp, acq, f, params)
    gpgo.run(max_iter = max_evals)
    mon.end()
    solX, solY = gpgo.getResult()

    return solX, -solY, mon