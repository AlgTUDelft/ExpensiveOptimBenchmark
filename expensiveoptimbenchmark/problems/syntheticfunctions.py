# Adapted from:
#
# -*- coding: utf-8 -*-
#==========================================
# Title:  syntheticFunctions.py
# Author: Binxin Ru and Ahsan Alvi
# Date:      20 August 2019
# Link:      https://arxiv.org/abs/1906.08878
#==========================================
# For license relevant to the original work,
# see /problems/solvers/CoCaBO/vendor/LICENSE

from .base import BaseProblem
import numpy as np
from scipy.optimize import rosen

# Wrapper

class MixedFunction(BaseProblem):
    def __init__(self, name, f, n_vars_d, n_vars_c, lbs, ubs, is_discrete_categorical, dolog=False):
        self.name = name
        self.f = f
        self.n_vars_d = n_vars_d
        self.n_vars_c = n_vars_c
        self.lb = lbs
        self.ub = ubs
        self.is_discrete_categorical = is_discrete_categorical
        self.d = n_vars_d + n_vars_c
        self.dolog=dolog
        assert len(lbs) == int(self.d), \
			f"Function {name} has a different number of lower bounds ({len(lbs)}) to its dimensionality ({self.d})"
        assert len(ubs) == int(self.d), \
			f"Function {name} has a different number of upper bounds ({len(ubs)}) to its dimensionality ({self.d})"
    
    def evaluate(self, x):
        if self.dolog:
            r = np.log(self.f(x) + 1)
        else:
            r = self.f(x)
        return float(r)

    def lbs(self):
        return self.lb

    def ubs(self):
        return self.ub

    def vartype(self):
        if type(self.is_discrete_categorical) is np.ndarray:
            assert len(self.is_discrete_categorical) == self.n_vars_d
            return np.array(['cat' if c else 'int' for c in self.is_discrete_categorical] + ['cont'] * self.n_vars_c)
        else:
            return np.array(['cat' if self.is_discrete_categorical else 'int'] * self.n_vars_d + ['cont'] * self.n_vars_c)

    def dims(self):
        return self.n_vars_d + self.n_vars_c

    def __str__(self):
        return f"MixedFunction(name={self.name},log={self.dolog})"


fns = []

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

SFhighdimRosenbrock = MixedFunction('highdimRosenbrock', highdimRosenbrock, 5, 20,
    np.concatenate([np.zeros(5), np.ones(20) * -2]),
    np.concatenate([np.ones(5), np.ones(20) * 2]),
    False)
SFhighdimRosenbrocklog = MixedFunction('highdimRosenbrocklog', highdimRosenbrock, 5, 20,
    np.concatenate([np.zeros(5), np.ones(20) * -2]),
    np.concatenate([np.ones(5), np.ones(20) * 2]),
    False, dolog=True)
fns.append(SFhighdimRosenbrock)
fns.append(SFhighdimRosenbrocklog)

#
def dim10Rosenbrock(x):
    # assert len(ht_list) == 3
    # h2 = [-2, -1, 0, 1, 2] #convert to these categories, as Cocabo assumes categories in (0,1,2,3,etc.)
    return rosen(x)/300 + 1e-6 * np.random.rand()

# SFDim10Rosenbrock = MixedFunction('dim10Rosenbrock', dim10Rosenbrock, 3, 10-3, 
#     np.concatenate([np.zeros(3), np.ones(10-3) * -2]),
#     np.concatenate([np.ones(3), np.ones(10-3) * 2]),
#     False)
# SFDim10Rosenbrocklog = MixedFunction('dim10Rosenbrocklog', dim10Rosenbrock, 3, 10-3, 
#     np.concatenate([np.zeros(3), np.ones(10-3) * -2]),
#     np.concatenate([np.ones(3), np.ones(10-3) * 2]),
#     False, dolog=True)
SFDim10Rosenbrock = MixedFunction('dim10Rosenbrock', dim10Rosenbrock, 3, 10-3, 
    np.ones(10) * -2,
    np.ones(10) * 2,
    False)
SFDim10Rosenbrocklog = MixedFunction('dim10Rosenbrocklog', dim10Rosenbrock, 3, 10-3, 
    np.ones(10) * -2,
    np.ones(10) * 2,
    False, dolog=True)
fns.append(SFDim10Rosenbrock)
fns.append(SFDim10Rosenbrocklog)

#
def dim53Rosenbrock(x):
    # assert len(ht_list) == 50
    # h2 = [1, 2] #convert to these categories, as Cocabo assumes categories in (0,1,2,3,etc.)
    return rosen(x)/20000 + 1e-6 * np.random.rand()

# SFDim53Rosenbrock = MixedFunction('dim53Rosenbrock', dim53Rosenbrock, 50, 3, 
#     np.concatenate([np.zeros(50), np.ones(3) * -2]),
#     np.concatenate([np.ones(50), np.ones(3) * 2]),
#     False)
# SFDim53Rosenbrocklog = MixedFunction('dim53Rosenbrocklog', dim53Rosenbrock, 50, 3, 
#     np.concatenate([np.zeros(50), np.ones(3) * -2]),
#     np.concatenate([np.ones(50), np.ones(3) * 2]),
#     False, dolog=True)
SFDim53Rosenbrock = MixedFunction('dim53Rosenbrock', dim53Rosenbrock, 50, 3, 
    np.ones(53) * -2,
    np.ones(53) * 2,
    False)
SFDim53Rosenbrocklog = MixedFunction('dim53Rosenbrocklog', dim53Rosenbrock, 50, 3, 
    np.ones(53) * -2,
    np.ones(53) * 2,
    False, dolog=True)
fns.append(SFDim53Rosenbrock)
fns.append(SFDim53Rosenbrocklog)

#
def dim238Rosenbrock(x):
    # assert len(ht_list) == 119
    # h2 = [-2, -1, 0, 1, 2] #convert to these categories, as Cocabo assumes categories in (0,1,2,3,etc.)
    return rosen(x)/50000 + 1e-6 * np.random.rand()

# SFDim238Rosenbrock = MixedFunction('dim238Rosenbrock', dim238Rosenbrock, 119, 119, 
#     np.concatenate([np.zeros(119), np.ones(119) * -2]),
#     np.concatenate([np.ones(119) * 4, np.ones(119) * 2]),
#     False)
# SFDim238Rosenbrocklog = MixedFunction('dim238Rosenbrocklog', dim238Rosenbrock, 119, 119, 
#     np.concatenate([np.zeros(119), np.ones(119) * -2]),
#     np.concatenate([np.ones(119) * 4, np.ones(119) * 2]),
#     False, dolog=True)
SFDim238Rosenbrock = MixedFunction('dim238Rosenbrock', dim238Rosenbrock, 119, 119, 
    np.ones(238) * -2,
    np.ones(238) * 2,
    False)
SFDim238Rosenbrocklog = MixedFunction('dim238Rosenbrocklog', dim238Rosenbrock, 119, 119, 
    np.ones(238) * -2,
    np.ones(238) * 2,
    False, dolog=True)
fns.append(SFDim238Rosenbrock)
fns.append(SFDim238Rosenbrocklog)


#
def cvxnonsep_psig20(x):
    # https://www.minlplib.org/cvxnonsep_psig20.html
    # convex MINLP test problem with non-separable signomial objective function
    # Objective at known optimum: 93.811388
    # First 10 variables are integer, the other 10 continuous
    return ((20000*x[0]**(-0.32)*x[1]**(-0.19)*x[2]**(-0.405)*x[3]**(-0.265)*x[4]**(-0.175)
    *x[5]**(-0.44)*x[6]**(-0.275)*x[7]**(-0.47)*x[8]**(-0.31)*x[9]**(-0.295)*x[10]**(-0.105)*
    x[11]**(-0.15)*x[12]**(-0.235)*x[13]**(-0.115)*x[14]**(-0.42)*x[15]**(-0.095)*
    x[16]**(-0.115)*x[17]**(-0.085)*x[18]**(-0.115)*x[19]**(-0.22))
    + np.sum(x))
     
SFCvxnonsep_psig20 = MixedFunction('cvxnonsep_psig20', cvxnonsep_psig20, 10, 10, 
    np.ones(20) * 1,
    np.ones(20) * 10,
    False)
fns.append(SFCvxnonsep_psig20)


def cvxnonsep_psig30(x):
    # https://www.minlplib.org/cvxnonsep_psig30.html
    # convex MINLP test problem with non-separable signomial objective function
    # Objective at known optimum: 78.99885434
    # First 15 variables are integer, the other 15 continuous
    return ((30000*x[0]**(-0.48)*x[1]**(-0.275)*x[2]**(-0.26)*x[3]**(-0.215)*x[4]**(-0.245)*
    x[5]**(-0.31)*x[6]**(-0.34)*x[7]**(-0.2)*x[8]**(-0.185)*x[9]**(-0.495)*x[10]**(-0.02)*
    x[11]**(-0.445)*x[12]**(-0.455)*x[13]**(-0.4)*x[14]**(-0.05)*x[15]**(-0.13)*
    x[16]**(-0.17)*x[17]**(-0.34)*x[18]**(-0.07)*x[19]**(-0.36)*x[20]**(-0.05)*
    x[21]**(-0.325)*x[22]**(-0.245)*x[23]**(-0.39)*x[24]**(-0.36)*x[25]**(-0.45)*
    x[26]**(-0.445)*x[27]**(-0.165)*x[28]**(-0.35)*x[29]**(-0.1))
    + np.sum(x))
     
SFCvxnonsep_psig30 = MixedFunction('cvxnonsep_psig30', cvxnonsep_psig30, 15, 15, 
    np.ones(30) * 1,
    np.ones(30) * 10,
    False)
fns.append(SFCvxnonsep_psig30)


def cvxnonsep_psig40(x):
    # https://www.minlplib.org/cvxnonsep_psig40.html
    # convex MINLP test problem with non-separable signomial objective function
    # Objective at known optimum: 85.49576764
    # First 20 variables are integer, the other 20 continuous
    return ((40000*x[0]**(-0.015)*x[1]**(-0.37)*x[2]**(-0.25)*x[3]**(-0.24)*x[4]**(-0.45)*
    x[5]**(-0.305)*x[6]**(-0.31)*x[7]**(-0.43)*x[8]**(-0.405)*x[9]**(-0.29)*x[10]**(-0.09)*
    x[11]**(-0.12)*x[12]**(-0.445)*x[13]**(-0.015)*x[14]**(-0.245)*x[15]**(-0.085)*
    x[16]**(-0.49)*x[17]**(-0.355)*x[18]**(-0.25)*x[19]**(-0.235)*x[20]**(-0.03)*
    x[21]**(-0.34)*x[22]**(-0.02)*x[23]**(-0.035)*x[24]**(-0.26)*x[25]**(-0.05)*
    x[26]**(-0.41)*x[27]**(-0.41)*x[28]**(-0.36)*x[29]**(-0.075)*x[29]**(-0.36)*
    x[30]**(-0.33)*x[31]**(-0.26)*x[32]**(-0.485)*x[33]**(-0.325)*x[34]**(-0.4)*
    x[35]**(-0.225)*x[36]**(-0.215)*x[37]**(-0.415)*x[38]**(-0.04)*x[39]**(-0.065))
    + np.sum(x))

     
SFCvxnonsep_psig40 = MixedFunction('cvxnonsep_psig40', cvxnonsep_psig40, 20, 20, 
    np.ones(40) * 1,
    np.ones(40) * 10,
    False)
fns.append(SFCvxnonsep_psig40)



#
def dim53Ackley(x):
    # assert len(ht_list) == 50
    # h2 = [0, 1] #convert to these categories, as Cocabo assumes categories in (0,1,2,3,etc.)
    a = 20
    b = 0.2
    c = 2*np.pi
    sum_sq_term = -a * np.exp(-b * np.sqrt(np.sum(np.square(x))/53))
    cos_term = -1*np.exp(np.sum(np.cos(c*np.copy(x))/53))
    result = a + np.exp(1) + sum_sq_term + cos_term
    return result + 1e-6 * np.random.rand()

SFDim53Ackley = MixedFunction('dim53Ackley', dim53Ackley, 50, 3, 
    np.concatenate([np.zeros(50), np.ones(3) * -1]),
    np.concatenate([np.ones(50), np.ones(3) * 1]),
    False)
fns.append(SFDim53Ackley)
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

    # assert len(ht_list) == 2
    ht1 = int(x[0])
    ht2 = int(x[1])

    if ht1 == 0:  # rosenbrock
        f = myrosenbrock(X)
    elif ht1 == 1:    # six hump
        f = mysixhumpcamp(X)
    elif ht1 == 2:    # beale
        f = mybeale(X)

    if ht2 == 0:  # rosenbrock
        f = f + myrosenbrock(X)
    elif ht2 == 1:    # six hump
        f = f + mysixhumpcamp(X)
    else:
        f = f + mybeale(X)

    y = f + 1e-6 * np.random.rand(f.shape[0], f.shape[1])
    return y.astype(float)

SFFunc2C = MixedFunction('func2C', func2C, 2, 2,
    np.asarray([0, 0, -1, -1]),
    np.asarray([2, 4, 1, 1]),
    True
)
fns.append(SFFunc2C)

#
def func3C(x):
    # ht is a categorical index
    # X is a continuous variable
    X = np.atleast_2d(x[3:])
    # assert len(ht_list) == 3
    ht1 = int(x[0])
    ht2 = int(x[1])
    ht3 = int(x[2])

    X = X * 2
    if ht1 == 0:  # rosenbrock
        f = myrosenbrock(X)
    elif ht1 == 1:    # six hump
        f = mysixhumpcamp(X)
    elif ht1 == 2:    # beale
        f = mybeale(X)

    if ht2 == 0:  # rosenbrock
        f = f + myrosenbrock(X)
    elif ht2 == 1:    # six hump
        f = f + mysixhumpcamp(X)
    else:
        f = f + mybeale(X)

    if ht3 == 0:  # rosenbrock
        f = f + 5 * mysixhumpcamp(X)
    elif ht3 == 1:    # six hump
        f = f + 2 * myrosenbrock(X)
    else:
        f = f + ht3 * mybeale(X)

    y = f + 1e-6 * np.random.rand(f.shape[0], f.shape[1])

    return y.astype(float)

SFFunc3C = MixedFunction('func3C', func3C, 3, 2,
    np.asarray([0, 0, 0, -1, -1]),
    np.asarray([2, 4, 3, 1, 1]), 
    True
)
fns.append(SFFunc3C)