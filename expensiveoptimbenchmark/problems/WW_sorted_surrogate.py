import os
import pickle
import numpy as np
import xgboost as xgb
import time

from .base import BaseProblem
from scipy.spatial.distance import cdist


# Define median-based ensemble of surrogates
# class medClassifier:
#     def __init__(self, classifiers=None):
#         self.classifiers = classifiers
#
#     def predict(self, X):
#         self.predictions_ = list()
#         for classifier in self.classifiers:
#             try:
#                 self.predictions_.append(classifier.predict(X)) #used for the random forest that is part of the ensemble
#             except:
#                 X = xgb.DMatrix(X)
#                 self.predictions_.append(classifier.predict(X)) #used for the XGBoost models that are part of the ensemble
#         med1 = np.median(self.predictions_, axis=0) #median of predictions
#         mean1 = np.mean(self.predictions_, axis=0) #mean of predictions
#         out = med1 + np.random.rand()*np.abs(med1-mean1) #add more noise if median is far from mean, indicating more uncertainty, also all noise is positive to focus on minimizing parts with more certainty
#         return out

class WW_sorted_surrogate(BaseProblem):
    def __init__(self, d, e):
        self.d = d
        # Load Ensemble
        # Ensemblefile = "Ensemble.pkl"
        # with open(Ensemblefile, 'rb') as file:
        #     self.Ensemble = pickle.load(file)

        # Instead of loading here, load Ensemble inside run_experiment.py and pass on to here
        #print('Loading Ensemble')
        self.Ensemble = e

    def to_np(self,xx):
        # transform to numpy array
        xx = np.asarray(xx)
        return xx

    def vec_to_2d(self,xx):
        # transform from vector values to 2D coordinates
        num_turbines = 5  # number of wind turbines
        xx = np.reshape(xx, (num_turbines, 2))
        return xx

    def sort2d(self,xx):
        # sort wind turbines from left to right
        # xx = np.transpose(xx) # so that x[0,:]
        ind = np.lexsort((xx[:, 1], xx[:, 0]))
        return xx[ind]

    def from_2d_to_vec(self,xx):
        # transform back from 2D coordinates to vector values
        num_turbines = 5  # number of wind turbines
        xx = np.reshape(xx, (1, num_turbines * 2))
        return xx[0]

    def sort2d_full(self,xx):
        # full transformation of the above functions
        return self.from_2d_to_vec(self.sort2d(self.vec_to_2d(self.to_np(xx))))

    #minimum distance constraint
    def constraint1(self,x):
        rotor_diameter = 126  # in meters
        farm_length = 333.33 * 5  # in meters
        coords = np.resize(x, (5, 2))
        min_dist = 999999  # minimum distance between turbines (Euclidean)

        for turb in range(4):
            dists = cdist([coords[turb]], coords[turb + 1:])
            next_min = np.min(dists)
            if next_min < min_dist:
                min_dist = next_min

        if min_dist * farm_length < 2 * rotor_diameter:
            constr = 0  # constraint not satisfied, wind turbines are too close to each other
        else:
            constr = 1  # constraint satisfied
        return constr

    def evaluate(self, x):
        assert len(x) == self.d
        x = np.array([x])
        if self.constraint1(x) == 0:
            return 0.0
        elif self.constraint1(x) == 1:
            x = self.sort2d_full(x[0])  # do sorting by x-coordinate
            x = [x]

            # take minimum over M queries to reduce noise
            #M = 5
            M = 1
            result_list = []


            for i in range(0,M):
                temp = self.Ensemble.predict(x)
                temp = temp[0]
                result_list.append(temp)
            result = np.min(result_list)
        #print(result_list)
        #print(result)
        else:
            print('Problem with evaluating the constraint.')
            result = None
        return result

    def lbs(self):
        return np.zeros(self.d, dtype=float)

    def ubs(self):
        return np.ones(self.d, dtype=float)

    def vartype(self):
        return np.array(['cont'] * self.dims())

    def dims(self):
        return self.d

    def __str__(self):
        return f"WW_sorted_surrogate(d={self.d})"



# Try the function


# x1 = [0.16583492632257624, 0.4871309875290023, 0.24153605985017246, 0.2954912222384124, 0.9558389075666868, 0.7992480932223422, 0.5400992985215289, 0.14902261675540462, 0.7592757901802544, 0.9983162571623986]
# t0 = time.time()
# print(evaluate_WWensemble(x1))
# t1 = time.time()
# print('time: ', t1 - t0)
#
# t0 = time.time()
# x2 = [0.4482183477205983, 0.027409116218383683, 0.25874372878562424, 0.6394764496282036, 1.0, 0.0, 1.0, 0.4692043394584843, 0.566542139621032, 0.0]
# t1 = time.time()
# print(evaluate_WWensemble(x2))
# print('time: ', t1 - t0)