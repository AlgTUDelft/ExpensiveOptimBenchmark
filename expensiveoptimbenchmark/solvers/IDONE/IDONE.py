### IDONE uses a piece-wise linear surrogate model for optimization of
### expensive cost functions with integer variables.
###
### IDONE_minimize(obj, x0, lb, ub, max_evals) solves the minimization problem
###
### min f(x)
### st. lb<=x<=ub, x is integer
###
### where obj is the objective function, x0 the initial guess,
### lb and ub are the bounds (assumed integer), 
### and max_evals is the maximum number of objective evaluations.

import os
import math
import random
import time
from sys import stdout
import numpy as np
from scipy.optimize import minimize, Bounds

def IDONE_minimize(obj, x0, lb, ub, max_evals, model_type, rand_evals=5, enable_scaling=False, verbose=1, log=False, sampling=None, exploration_prob = None):
	d = len(x0) # dimension, number of variables
	current_time = time.time() # time when starting the algorithm
	next_X = [] # candidate solution presented by the algorithm
	rng = np.random.default_rng()

	# Compute exploration probability
	if exploration_prob is None: # Optional to pick probability of making a random step
		prob = 1/d
	else:
		prob = exploration_prob

	# Decide sampling strategy
	sampling_strategies = ['none', 'thompson', 'uniform'] # Allowed sampling strategies
	if sampling in sampling_strategies or sampling is None:
		if sampling == 'thompson':
			thompson_sampling = True
			uniform_sampling = False
		elif sampling == 'uniform':
			thompson_sampling = False
			uniform_sampling = True
		else:
			thompson_sampling = False
			uniform_sampling = False
	else:
		raise ValueError(f"Invalid sampling argument given. Change to an existing one: {sampling_strategies}")
	
	## Initialize the surrogate model
	def initializeModel():
		next_X = np.copy(x0)
		
		# def ReLU(x): # Rectified Linear Unit
		# 	temp = np.copy(x)
		# 	for i, xx in enumerate(x):
		# 		temp[i] = [max(0,xx[0])]
		# 	return np.asarray(temp)
		def ReLU(x):
			return np.maximum(0, x)

		# def ReLUderiv(x): # Derivative of Rectified Linear Unit
		# 	temp = np.copy(x)
		# 	for i, xx in enumerate(x):
		# 		temp[i] = [0.5*(xx[0]==0) + (xx[0]>0)]
		# 	return np.asarray(temp)
		def ReLUderiv(x):
			return (x > 0) + 0.5 * (x == 0)
		
		# Define basis functions Z=ReLU(W*x+B)
		W = [] # Basis function weights
		B = [] # Basis function bias
		
		# Add a constant basis function independent on the variable x, giving the model an offset
		W.append([0]*d)
		B.append([1])
		
		# Add basis functions dependent on one variable
		for k in range(d): 
			for i in range(int(lb[k]),int(ub[k])+1):
				if i == lb[k]:
					temp = [0]*d
					temp[k] = 1
					W.append(np.copy(temp))
					B.append([-i])
				elif i == ub[k]:
					temp = [0]*d
					temp[k] = -1
					W.append(np.copy(temp))
					B.append([i])
				else:
					temp = [0]*d
					temp[k] = -1
					W.append(np.copy(temp))
					B.append([i])
					temp = [0]*d
					temp[k] = 1
					W.append(np.copy(temp))
					B.append([-i])
					
		# Add basis functions dependent on two subsequent variables
		if model_type == 'advanced':
			for k in range(1,d): 
				for i in range(int(lb[k]-ub[k-1]),int(ub[k]-lb[k-1])+1):
					if i == lb[k]-ub[k-1]:
						temp = [0]*d
						temp[k] = 1
						temp[k-1] = -1
						W.append(np.copy(temp))
						B.append([-i])
					elif i == ub[k]-lb[k-1]:
						temp = [0]*d
						temp[k] = -1
						temp[k-1] = 1
						W.append(np.copy(temp))
						B.append([i])
					else:
						temp = [0]*d
						temp[k] = -1
						temp[k-1] = 1
						W.append(np.copy(temp))
						B.append([i])
						temp = [0]*d
						temp[k] = 1
						temp[k-1] = -1
						W.append(np.copy(temp))
						B.append([-i])
				
		W = np.asarray(W)	
		B = np.asarray(B)
		
		# Transform input into model basis functions Z=ReLU(W*x+B)
		def Z(x): 
			x = np.asarray(x)
			x = x.reshape(-1,1)
			temp = np.matmul(W,x)
			temp = temp + B
			temp = np.asarray(temp)
			Zx = ReLU(temp)
			return Zx
		
		# Derivative of basis functions w.r.t. x
		def Zderiv(x):
			x = np.asarray(x)
			x = x.reshape(-1,1)
			temp = np.matmul(W,x)
			temp = temp +B
			temp = np.asarray(temp)
			dZx = ReLUderiv(temp)
			return dZx
		
		D = len(B) # Number of basis functions
		c = np.ones((D,1)) # Model weights, to be trained with recursive least squares (RLS)
		c[0] = 0 # Start out with no model offset
		reg = 1e-3 # Regularization parameter. 1e-8 is good for the noiseless case, change to something like 1e-3 if there is noise.
		P = np.diag(np.ones(D))/reg # RLS covariance matrix
		model = {'W':W, 'B':B, 'ReLU':ReLU, 'ReLUderiv':ReLUderiv, 'Z':Z, 'Zderiv':Zderiv, 'D':D, 'c':c, 'reg':reg, 'P':P} # Store model variables in a dictionary
		
		return next_X, model
	
	## Update the model when a new data point (x,y) comes in
	def updateModel(x,y, model): 
		Zx = model['Z'](x)			
		# Recursive least squares algorithm
		temp = np.matmul(model['P'], Zx)
		g = temp/(1+np.matmul(np.transpose(Zx),temp))
		model['P'] = model['P'] - np.matmul(g, np.transpose(temp))
		model['c'] = model['c'] + ( y-np.matmul( np.transpose(Zx), model['c'] ) ) * g # Only here, output y is used (to update the model weights)
				
		# Define model output for any new input x2
		def out(x2):
			if thompson_sampling is True:
				# If Thompson Sampling is performed, compute output by sampled weights c_samp
				return np.matmul( np.transpose(model['c_samp']), model['Z'](x2) ).item(0,0)
			else:
				return np.matmul( np.transpose(model['c']), model['Z'](x2) ).item(0,0)
		
		# Define model output derivative for any new input x2 (used in the optimization step)
		def deriv(x2):
			if thompson_sampling is True:
				c = np.transpose(model['c_samp'])
			else:
				c = np.transpose(model['c'])			
			# temp = np.reshape(model['Zderiv'](x2),(model['Zderiv'](x2)).shape[0])
			temp = np.reshape(model['Zderiv'](x2), -1)
			temp = np.matmul(np.diag(temp), model['W'])
			result = np.transpose(np.matmul( c, temp  ))
			return result

		model['out'] = out
		model['outderiv'] = deriv
		return model
	
	
	
	### Start actual algorithm
	next_X, model = initializeModel()
	best_X = np.copy(next_X) # Best candidate solution found so far
	best_y = 9999999 # Best objective function value found so far
	
	## Iteratively evaluate the objective, update the model, find the minimum of the model, and explore the search space
	for ii in range(0,max_evals):
		if verbose > 0 and ii % verbose == 0:
			stdout.write(f"\rStarting IDONE iteration {ii}/{max_evals}")
			stdout.flush()
		x = np.copy(next_X).astype(int)
		if ii==0:
			y = obj(x) # Evaluate the objective
			
			# Scale with respect to initial y value, causing the optimum to lie below 0.
			# This is better for exploitation and prevents the algorithm from getting stuck at the boundary.
			y0 = y
			def scale(y):
				if enable_scaling:
					if abs(y0)>1e-8:
						y = (y-y0)/abs(y0)
					else:
						y = (y-y0)
				return y
			def inv_scale(y):
				if enable_scaling:
					if abs(y0)>1e-8:
						y = y*abs(y0)+y0
					else:
						y = y+y0
				return y
			
			y = scale(y)
		else:
			y = obj(x) # Evaluate the objective
			y = scale(y)
		
		
		# Keep track of the best found objective value and candidate solution so far
		if y < best_y:
			best_X = np.copy(x)
			best_y = y
			
			
		## Update the surrogate model
		time_start = time.time() 
		model = updateModel(x,y, model)
		update_time = time.time()-time_start # Time used to update the model

		# Perform sampling of weights
		if thompson_sampling is True:
			model['c_samp'] = np.random.multivariate_normal(model['c'].T[0], model['P']).T
			model['c_samp'] = np.reshape(model['c_samp'], (model['D'], 1))
		
		# Should the next sample be chosen using the surrogate model?
		minimization_time = 0.0
		next_X_before_exploration = x
		if ii >= rand_evals - 1:
			## Minimization of the surrogate model
			time_start = time.time()
			temp = minimize(model['out'], x, method='L-BFGS-B', bounds = Bounds(lb, ub), jac=model['outderiv'], options={'maxiter':20,'maxfun':20})
			minimization_time = time.time()-time_start # Time used to find the minimum of the model
			next_X = np.copy(temp.x)
			next_X = np.round(next_X) # Round to nearest integer point
			next_X = [int(x) for x in next_X]

			
			# Just to be sure, clip to the bounds
			np.clip(next_X, lb, ub)
			
			# Check if minimizer really gives better result
			if model['out'](next_X) > model['out'](x) + 1e-8:
				print('Warning: minimization of the surrogate model in IDONE yielded a worse solution, maybe something went wrong.')
			
			## Exploration step (else the algorithm gets stuck in the local minimum of the surrogate model)
			next_X_before_exploration = np.copy(next_X)
			next_X = np.copy(next_X)

			# Skip exploration before the last iteration, to end at the exact minimum of the surrogate model.
			# or skip random exploration if Thompson sampling is being used
			if ii<max_evals-2 and thompson_sampling is not True:
				if uniform_sampling is not True:
					for j in range(0,d):
						r = rng.random()
						a = next_X[j]
						if r < prob:
							if a==lb[j] and a<ub[j]:
								a += 1 # Explore to the right
							elif a==ub[j] and a>lb[j]:
								a -= 1 # Explore to the left
							elif a>lb[j] and a<ub[j]:
								r2 = rng.random() # Explore left or right
								if r2<0.5:
									a += 1
								else:
									a -= 1
						next_X[j]=a
				else:
					mutated = rng.uniform(size=(d,)) < prob
					next_X[mutated] = rng.integers(lb[mutated], ub[mutated], endpoint=True)

			
			# Just to be sure, clip to the bounds again
			np.clip(next_X, lb, ub)
		else:
			# Random sample otherwise!
			next_X = rng.integers(lb, ub, endpoint=True)
		
		# If even after exploration x does not change, go to a completely random x
		#if np.allclose(next_X,x):
		#	next_X = np.round(np.random.rand(d)*(ub-lb) + lb)
		
		# Save data to log file
		filename = 'log_IDONE_'+ str(current_time) + ".log"
		if log:
			with open(filename, 'a') as f:
				print('\n\n IDONE iteration: ', ii, file=f)
				print('Time spent training the model:				 ', update_time, file=f)
				print('Time spent finding the minimum of the model: ', minimization_time, file=f)
				print('Current time: ', time.time(), file=f)
				print('Evaluated data point and evaluation:						   ', np.copy(x).astype(int),  ', ',  inv_scale(y), file=f)
				print('Best found data point and evaluation so far:				   ', np.copy(best_X).astype(int),  ', ',  inv_scale(best_y), file=f)
				print('Best data point according to the model and predicted value:    ', next_X_before_exploration, ', ', inv_scale(model['out'](next_X_before_exploration)), file=f)
				print('Suggested next data point and predicted value:			       ', next_X,   ', ',  inv_scale(model['out'](next_X)), file=f)
				if ii>=max_evals-1:
					print('Model parameters: ', np.transpose(model['c']), file=f)
	
	return best_X, inv_scale(best_y), model, filename


# Read data from log file (this reads the best found objective values at each iteration)
def read_log(filename):
	with open(filename,'r') as f:
			IDONEfile = f.readlines()
			IDONE_best = []
			for i, lines in enumerate(IDONEfile):
				searchterm = 'Best data point according to the model and predicted value:'
				if searchterm in lines:
					temp = IDONEfile[i-1]
					temp = temp.split('] , ')
					temp = temp[1]
					IDONE_best.append(float(temp))
	return IDONE_best

# Plot the best found objective values at each iteration
def plot_results(filename):
	import matplotlib.pyplot as plt
	IDONE_ev=read_log(filename)
	plt.plot(IDONE_ev)
	plt.xlabel('Iteration')
	plt.ylabel('Objective')
	plt.grid()
	plt.show()