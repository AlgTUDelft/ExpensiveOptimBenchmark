import os
import pickle
import numpy as np
import xgboost as xgb
import time

from .base import BaseProblem



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

class WWsurrogate(BaseProblem):
    def __init__(self, d, e):
        self.d = d
        # Load Ensemble
        # Ensemblefile = "Ensemble.pkl"
        # with open(Ensemblefile, 'rb') as file:
        #     self.Ensemble = pickle.load(file)

        # Instead of loading here, load Ensemble inside run_experiment.py and pass on to here
        print('Loading Ensemble')
        self.Ensemble = e

    def evaluate(self, x):
        assert len(x) == self.d
        x = np.array([x])
        result = self.Ensemble.predict(x)
        return result[0]

    def lbs(self):
        return np.zeros(self.d, dtype=float)

    def ubs(self):
        return np.ones(self.d, dtype=float)

    def vartype(self):
        return np.array(['cont'] * self.dims())

    def dims(self):
        return self.d

    def __str__(self):
        return f"WWsurrogate(d={self.d})"



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