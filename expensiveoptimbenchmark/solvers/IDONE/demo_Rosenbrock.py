import IDONE
import numpy as np
import os
from scipy.optimize import rosen

def test_Rosenbrock(d):
	print(f"Testing IDONE on the {d}-dimensional Rosenbrock function with integer constraints.")
	print("The known global minimum is f(1,1,...,1)=0")
	lb = -5*np.ones(d).astype(int) # Lower bound
	ub = 10*np.ones(d).astype(int) # Upper bound
	x0 = np.round(np.random.rand(d)*(ub-lb) + lb) # Random initial guess
	max_evals = 500 # Maximum number of IDONE iterations
	def f(x):
		scaling = d*(100*((ub[0]-lb[0]**2)**2)+(ub[0]-1)**2)
		result = rosen(x)/scaling
		return result
	solX, solY, model, logfile = IDONE.IDONE_minimize(f, x0, lb, ub, max_evals)
	print("Solution found: ")
	print(f"X = {solX}")
	print(f"Y = {solY}")
	return solX, solY, model, logfile


d = 5 # Change this number to optimize the Rosenbrock function for different numbers of variables
solX, solY, model, logfile = test_Rosenbrock(d)

# Visualise the results
IDONE.plot_results(logfile)