import sys
import numpy as np
from problems.convex import Convex

from solvers.IDONE.wIDONE import optimize_IDONE
from solvers.pyGPGO.wpyGPGO import optimize_pyGPGO
from solvers.hyperopt.whyperopt import optimize_hyperopt_tpe

d = 50
problem = Convex(d, 0)
max_evals = 100

## IDONE
solX, solY, mon = optimize_IDONE(problem, max_evals)
solXID, solYID, monID = solX, solY, mon


print(f"IDONE found solution {solX} with {solY}.")
print(f"Spent {np.mean(mon.model_time())}s per call on deciding the next point,")
print(f"and {np.mean(mon.eval_time())}s per call on the evaluation itself.")
print(f"Optimum was {problem.x_star}")

# Original:
# IDONE found solution [ 7 -1  7] with 3.7441506871558685.
# Spent 0.09994747764185856s per call on deciding the next point,
# and 0.0002001047134399414s per call on the evaluation itself.

# Patchy:
# IDONE found solution [-2  3  9] with 0.001668069025581979.
# Spent 0.013736900530363383s per call on deciding the next point,
# and 0.0002500653266906738s per call on the evaluation itself.

## pyGPGO - 1
# from pyGPGO.covfunc import matern32
# from pyGPGO.acquisition import Acquisition
# from pyGPGO.surrogates.GaussianProcess import GaussianProcess
# from pyGPGO.GPGO import GPGO

# cov = matern32()
# gp = GaussianProcess(cov)
# # gp = GaussianProcess(cov, optimize=True)
# acq = Acquisition(mode='ExpectedImprovement')

# solX, solY, mon = optimize_pyGPGO(problem, max_evals, gp, acq)
# solXGPGO, solYGPGO, monGPGO = solX, solY, mon

# print(f"pyGPGO/1 found solution {solX} with {solY}")
# print(f"Spent {np.mean(mon.model_time())}s per call on building a model,")
# print(f"and {np.mean(mon.eval_time())}s per call on the evaluation itself.")

##
solX, solY, mon = optimize_hyperopt_tpe(problem, max_evals)
solXHyp, solYHyp, monHyp = solX, solY, mon


print(f"hyperopt found solution {solX} with {solY}.")
print(f"Spent {np.mean(mon.model_time())}s per call on deciding the next point,")
print(f"and {np.mean(mon.eval_time())}s per call on the evaluation itself.")

## pyGPGO - 2, only works on Unix/Linux. Fails horribly on windows.

# if sys.platform != "win32":
#     from pyGPGO.surrogates.GaussianProcessMCMC import GaussianProcessMCMC
#     import pymc3 as pm

#     gp = GaussianProcessMCMC(cov, niter=300, burnin=100, step=pm.Slice)
#     acq = Acquisition(mode='IntegratedExpectedImprovement')

#     solX, solY, mon = optimize_pyGPGO(problem, max_evals, gp, acq)
#     solXGPGO2, solYGPGO2, monGPGO2 = solX, solY, mon

#     print(f"pyGPGO/2 found solution {solX} with {solY}")
#     print(f"Spent {np.mean(mon.model_time())}s per call on building a model,")
#     print(f"and {np.mean(mon.eval_time())}s per call on the evaluation itself.")

## Plot!
import matplotlib.pyplot as plt

plt.plot(monID.model_time(), label="IDONE")
# plt.plot(monGPGO.model_time(), label="GPGO/1")
plt.plot(monHyp.model_time(), label="HyperOpt")
# plt.plot(monGPGO2.model_time(), label="GPGO/2")
plt.legend()
plt.show()