import os
import pickle
import numpy as np
import xgboost as xgb

# This surrogate is trained on windwake data that was first sorted by x-coordinate, to reduce permutation invariance.
# The surrogate also first sorts any input by x-coordinate.
# y-coordinates are used in case of ties.



def to_np(xx):
    # transform to numpy array
    xx = np.asarray(xx)
    return xx

def vec_to_2d(xx):
    # transform from vector values to 2D coordinates
    num_turbines = 5  # number of wind turbines
    xx = np.reshape(xx, (2, num_turbines))
    xx = np.transpose(xx)
    return xx

def sort2d(xx):
    # sort wind turbines from left to right
    ind = np.lexsort((xx[:, 1], xx[:, 0]))
    return xx[ind]

def from_2d_to_vec(xx):
    # transform back from 2D coordinates to vector values
    num_turbines = 5  # number of wind turbines
    xx = np.concatenate((np.reshape(xx[:,0], (1,num_turbines)), np.reshape(xx[:,1], (1,num_turbines))),axis=1)
    return xx[0]

def sort2d_full(xx):
    # full transformation of the above functions
    return from_2d_to_vec(sort2d(vec_to_2d(to_np(xx))))

# Define median-based ensemble of surrogates
class medClassifier:
    def __init__(self, classifiers=None):
        self.classifiers = classifiers

    def predict(self, X):
        X = sort2d_full(X[0]) #do sorting by x-coordinate
        X = [X]
        self.predictions_ = list()
        for classifier in self.classifiers:
            try:
                self.predictions_.append(classifier.predict(X)) #used for the random forest that is part of the ensemble
            except:
                X = xgb.DMatrix(X)
                self.predictions_.append(classifier.predict(X)) #used for the XGBoost models that are part of the ensemble
        med1 = np.median(self.predictions_, axis=0) #median of predictions
        mean1 = np.mean(self.predictions_, axis=0) #mean of predictions
        noise_param = 1 # Noise parameter
        out = med1 + noise_param*np.random.rand()*np.abs(med1-mean1) #add more noise if median is far from mean, indicating more uncertainty, also all noise is positive to focus on minimizing parts with more certainty
        return out

# Load Ensemble
#Ensemblefile = os.path.join(folder_path,"Ensemble.pkl")
Ensemblefile = "Ensemble_sorted.pkl"
with open(Ensemblefile, 'rb') as file:
    Ensemble = pickle.load(file)


def evaluate_WWensemble(x):
    x = np.array([x])
    pred = Ensemble.predict(x)
    return pred[0]

# Try the function

x1 = [0.16583492632257624, 0.4871309875290023, 0.24153605985017246, 0.2954912222384124, 0.9558389075666868, 0.7992480932223422, 0.5400992985215289, 0.14902261675540462, 0.7592757901802544, 0.9983162571623986]
print(evaluate_WWensemble(x1))

x2 = [0.4482183477205983, 0.027409116218383683, 0.25874372878562424, 0.6394764496282036, 1.0, 0.0, 1.0, 0.4692043394584843, 0.566542139621032, 0.0]
print(evaluate_WWensemble(x2))

# good solution of about 71 GWh
x3 = [0.021946399792690363,0.14758613294341658, 0.5527813580504545, 0.8365883772315181,0.9889926312512337, 0.9935644172435926, 0.07890115878277747,0.2889801154078123,0.9925503140918277,4148795881645418]
print(evaluate_WWensemble(x3))

x4 = [np.float64(0.5561898390342438), np.float64(0.8006886197578583), np.float64(0.24212881638026462), np.float64(0.5614968582664879), np.float64(0.8195496761384983), np.float64(0.7511574305317662), np.float64(0.7398266355358496), np.float64(0.22941958927880024), np.float64(0.5207337011354721), np.float64(0.9267783983342328)]
print(evaluate_WWensemble(x4))

#74GWh in simulator:
x5 = [0.039602075085971904, 0.9345808188563574, 0.10123332612146797, 0.5039334697500139, 0.9883610447867386, 0.24460605636236385, 0.04091102244130369, 0.9675233394185299, 0.12457466692504467, 0.9958827777647239]
print(evaluate_WWensemble(x5))