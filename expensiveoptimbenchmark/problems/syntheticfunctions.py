# Adapted from:
#
# -*- coding: utf-8 -*-
#==========================================
# Title:  syntheticFunctions.py
# Author: Binxin Ru and Ahsan Alvi
# Date:	  20 August 2019
# Link:	  https://arxiv.org/abs/1906.08878
#==========================================
# For license relevant to the original work,
# see /problems/solvers/CoCaBO/vendor/LICENSE

import numpy as np
from scipy.optimize import rosen

# Wrapper

class MixedFunction:
	def __init__(name, f, n_vars_d, n_vars_cont, lbs, ubs):
		self.name = name
		self.f = f
		self.n_vars_d = n_vars_d
		self.n_vars_cont = n_vars_cont
		self.lb = lbs
		self.ub = ubs
		self.d = n_vars_d + n_vars_cont
		assert len(self.lbs) == self.d
		assert len(self.ubs) == self.d
	
	def evaluate(self, x):
        return self.f(x)

    def lbs(self):
        return self.lb

    def ubs(self):
        return self.ub

    def vartype(self):
        return np.array(['int'] * self.n_vars_d + ['cont'] * self.n_vars_c)

    def dims(self):
        return self.n_vars_d + self.n_vars_c

    def __str__(self):
        return f"MixedFunction(name={self.name})"

 

# =============================================================================
# Rosenbrock Function (f_min = 0)
# https://www.sfu.ca/~ssurjano/rosen.html
# =============================================================================
def myrosenbrock(X):
	X = np.asarray(X)
	X = X.reshape((-1, 2))
	if len(X.shape) == 1:  # one observation
		x1 = X[0]
		x2 = X[1]
	else:  # multiple observations
		x1 = X[:, 0]
		x2 = X[:, 1]
	fx = 100 * (x2 - x1 ** 2) ** 2 + (x1 - 1) ** 2
	return fx.reshape(-1, 1) / 300

#Adapted
def highdimRosenbrock(x):
	# assert len(ht_list) == 5
	# h2 = [-2, -1, 0, 1, 2] #convert to these categories, as Cocabo assumes categories in (0,1,2,3,etc.)
	return rosen(x)/300 + 1e-6 * np.random.rand()

SFhighdimRosenbrock = MixedFunction(highdimRosenbrock, 5, 20,
	np.concatenate([np.zeros(5), np.ones(20) * -2]),
	np.concatenate([np.ones(5), np.ones(20) * 2]))

#
def dim10Rosenbrock(x):
	# assert len(ht_list) == 3
	# h2 = [-2, -1, 0, 1, 2] #convert to these categories, as Cocabo assumes categories in (0,1,2,3,etc.)
	return rosen(x)/300 + 1e-6 * np.random.rand()

SFDim10Rosenbrock = MixedFunction(dim10Rosenbrock, 3, 10-3, 
	np.concatenate([np.zeros(3), np.ones(10-3) * -2]),
	np.concatenate([np.ones(4), np.ones(10-3) * 2]))

#
def dim53Rosenbrock(x):
	# assert len(ht_list) == 50
	# h2 = [1, 2] #convert to these categories, as Cocabo assumes categories in (0,1,2,3,etc.)
	return rosen(x)/20000 + 1e-6 * np.random.rand()

SFDim53Rosenbrock = MixedFunction(dim53Rosenbrock, 50, 3, 
	np.concatenate([np.zeros(50), np.ones(3) * -2]),
	np.concatenate([np.ones(50), np.ones(3) * 2]))

#
def dim238Rosenbrock(x):
	# assert len(ht_list) == 119
	# h2 = [-2, -1, 0, 1, 2] #convert to these categories, as Cocabo assumes categories in (0,1,2,3,etc.)
	return rosen(x)/50000 + 1e-6 * np.random.rand()

SFDim238Rosenbrock = MixedFunction(dim238Rosenbrock, 119, 119, 
	np.concatenate([np.zeros(119), np.ones(119) * -2]),
	np.concatenate([np.ones(119) * 4, np.ones(119) * 2]))

#
def dim53Ackley(x):
	# assert len(ht_list) == 50
	# h2 = [0, 1] #convert to these categories, as Cocabo assumes categories in (0,1,2,3,etc.)
	a = 20
	b = 0.2
	c = 2*np.pi
	sum_sq_term = -a * np.exp(-b * np.sqrt(np.sum(np.square(XX))/53))
	cos_term = -1*np.exp(np.sum(np.cos(c*np.copy(XX))/53))
	result = a + np.exp(1) + sum_sq_term + cos_term
	return result + 1e-6 * np.random.rand()

SFDim53Ackley = MixedFunction(dim53Ackley, 50, 3, 
	np.concatenate([np.zeros(50), np.ones(3) * -1]),
	np.concatenate([np.ones(50), np.ones(3) * 1]))
#/Adapted


# =============================================================================
#  Six-hump Camel Function (f_min = - 1.0316 )
#  https://www.sfu.ca/~ssurjano/camel6.html		  
# =============================================================================
def mysixhumpcamp(X):
	X = np.asarray(X)
	X = np.reshape(X, (-1, 2))
	if len(X.shape) == 1:
		x1 = X[0]
		x2 = X[1]
	else:
		x1 = X[:, 0]
		x2 = X[:, 1]
	term1 = (4 - 2.1 * x1 ** 2 + (x1 ** 4) / 3) * x1 ** 2
	term2 = x1 * x2
	term3 = (-4 + 4 * x2 ** 2) * x2 ** 2
	fval = term1 + term2 + term3
	return fval.reshape(-1, 1) / 10

# =============================================================================
# Beale function (f_min = 0)
# https://www.sfu.ca/~ssurjano/beale.html
# =============================================================================
def mybeale(X):
	X = np.asarray(X) / 2
	X = X.reshape((-1, 2))
	if len(X.shape) == 1:
		x1 = X[0] * 2
		x2 = X[1] * 2
	else:
		x1 = X[:, 0] * 2
		x2 = X[:, 1] * 2
	fval = (1.5 - x1 + x1 * x2) ** 2 + (2.25 - x1 + x1 * x2 ** 2) ** 2 + (
			2.625 - x1 + x1 * x2 ** 3) ** 2
	return fval.reshape(-1, 1) / 50


def func2C(x):
	# ht is a categorical index
	# X is a continuous variable
	X = x[2:]

	assert len(ht_list) == 2
	ht1 = ht_list[0]
	ht2 = ht_list[1]

	if ht1 == 0:  # rosenbrock
		f = myrosenbrock(X)
	elif ht1 == 1:	# six hump
		f = mysixhumpcamp(X)
	elif ht1 == 2:	# beale
		f = mybeale(X)

	if ht2 == 0:  # rosenbrock
		f = f + myrosenbrock(X)
	elif ht2 == 1:	# six hump
		f = f + mysixhumpcamp(X)
	else:
		f = f + mybeale(X)

	y = f + 1e-6 * np.random.rand(f.shape[0], f.shape[1])
	return y.astype(float)

SFFunc2C = MixedFunction(func2C, 2, 2,
	np.asarray([0, 0, -1, -1]),
	np.asarray([2, 4, 1, 1])
)

#
def func3C(x):
	# ht is a categorical index
	# X is a continuous variable
	X = np.atleast_2d(x[3:])
	assert len(ht_list) == 3
	ht1 = ht_list[0]
	ht2 = ht_list[1]
	ht3 = ht_list[2]

	X = X * 2
	if ht1 == 0:  # rosenbrock
		f = myrosenbrock(X)
	elif ht1 == 1:	# six hump
		f = mysixhumpcamp(X)
	elif ht1 == 2:	# beale
		f = mybeale(X)

	if ht2 == 0:  # rosenbrock
		f = f + myrosenbrock(X)
	elif ht2 == 1:	# six hump
		f = f + mysixhumpcamp(X)
	else:
		f = f + mybeale(X)

	if ht3 == 0:  # rosenbrock
		f = f + 5 * mysixhumpcamp(X)
	elif ht3 == 1:	# six hump
		f = f + 2 * myrosenbrock(X)
	else:
		f = f + ht3 * mybeale(X)

	y = f + 1e-6 * np.random.rand(f.shape[0], f.shape[1])

	return y.astype(float)

SFFunc3C = MixedFunction(func3C, 3, 2,
	np.asarray([0, 0, 0, -1, -1]),
	np.asarray([2, 4, 3, 1, 1])
)