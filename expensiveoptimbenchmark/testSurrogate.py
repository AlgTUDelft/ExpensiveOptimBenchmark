import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns; sns.set_style(style='white') #sns.set()
from sklearn.metrics import mean_squared_error, max_error, mean_absolute_error
from sklearn import svm
from sklearn.model_selection import train_test_split
import time
import pickle
import xgboost as xgb

# Import data
def load_data(folder_path):
    # Loads all iter.csv files in the folder
    log_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    file_list = [f for f in log_files if f.endswith('iters.csv')]
    # Read log files
    iter_df = pd.DataFrame()
    for f in file_list:
        temp = pd.read_csv(f)
        iter_df = pd.concat([iter_df,temp])
    return iter_df



def get_data_all(iter_df):
    df = iter_df
    #df_train, df_test = train_test_split(df, test_size=0.0)
    df_train = df
    y_train = df_train['iter_fitness']
    y_train = y_train.to_numpy(dtype=float)
    y_train = y_train*1e-9 # for windwake problem, to convert from Wh to GWh
    x_train = df_train['iter_x']
    x_train = x_train.to_numpy()
    temp = np.array([])
    for x in x_train:
        #x = list(map(str.strip, x.strip('][').replace('"', '').split(',')))
        x = x.strip('[]')
        x = np.fromstring(x,dtype=float,sep=',')
        if temp.size==0:
            temp = np.copy(x)
        else:
            temp = np.vstack((temp,x))
    #x_all = x_all.to_numpy(dtype=float)
    #x_all= np.stack(x_all,axis=0)
    x_train = np.copy(temp)

    # y_test = df_test['iter_fitness']
    # y_test = y_test.to_numpy(dtype=float)
    # y_test = y_test*1e-9 # for windwake problem, to convert from Wh to GWh
    # x_test = df_test['iter_x']
    # x_test = x_test.to_numpy()
    # temp2 = np.array([])
    # for x in x_test:
    #     #x = list(map(str.strip, x.strip('][').replace('"', '').split(',')))
    #     x = x.strip('[]')
    #     x = np.fromstring(x,dtype=float,sep=',')
    #     if temp2.size==0:
    #         temp2 = np.copy(x)
    #     else:
    #         temp2 = np.vstack((temp2,x))
    # #x_all = x_all.to_numpy(dtype=float)
    # #x_all= np.stack(x_all,axis=0)
    # x_test = np.copy(temp2)



    print('Train Data labels:', y_train.shape)
    print('Train Data inputs:', x_train.shape)
    # print('Test Data labels:', y_test.shape)
    # print('Test Data inputs:', x_test.shape)



    return x_train, y_train#, x_test, y_test




# Load surrogates
folder_path = r"C:\Users\20205209\OneDrive - TU Eindhoven\TUE\Code\BO Summer School Hasselt\BO windwake lab\BO_lab\finished models"
RFfile = os.path.join(folder_path,"RF.pkl")
with open(RFfile, 'rb') as file:
    RF = pickle.load(file)

XGB6file = os.path.join(folder_path,"XGB6.pkl")
with open(XGB6file, 'rb') as file:
    XGB6 = pickle.load(file)


XGB18file = os.path.join(folder_path,"XGB18.pkl")
with open(XGB18file, 'rb') as file:
    XGB18 = pickle.load(file)

XGB60file = os.path.join(folder_path,"XGB60.pkl")
with open(XGB60file, 'rb') as file:
    XGB60 = pickle.load(file)

XGB240file = os.path.join(folder_path,"XGB240.pkl")
with open(XGB240file, 'rb') as file:
    XGB240 = pickle.load(file)

class medClassifier:
    def __init__(self, classifiers=None):
        self.classifiers = classifiers

    # def fit(self, X, y):
    #     for classifier in self.classifiers:
    #         classifier.fit(X, y)

    def predict(self, X):
        self.predictions_ = list()
        for classifier in self.classifiers:
            try:
                self.predictions_.append(classifier.predict(X))
            except:
                X = xgb.DMatrix(X)
                self.predictions_.append(classifier.predict(X))
        med1 = np.median(self.predictions_, axis=0) #median of predictions
        mean1 = np.mean(self.predictions_, axis=0) #mean of predictions
        out = med1 + np.random.rand()*np.abs(med1-mean1) #add more noise if median is far from mean, indicating more uncertainty, also all noise is positive to focus on minimizing parts with more certainty
        return out

# Create Ensemble
# Ensemble = medClassifier([RF, XGB6, XGB18, XGB60, XGB240])

# Save Ensemble
# Ensemblefile = "Ensemble.pkl"
# with open(Ensemblefile, 'wb') as file:
#     pickle.dump(Ensemble, file)

# Load Ensemble
Ensemblefile = os.path.join(folder_path,"Ensemble.pkl")
with open(Ensemblefile, 'rb') as file:
    Ensemble = pickle.load(file)

smalldata = 1
if smalldata:
    x1 = [0.16583492632257624, 0.4871309875290023, 0.24153605985017246, 0.2954912222384124, 0.9558389075666868, 0.7992480932223422, 0.5400992985215289, 0.14902261675540462, 0.7592757901802544, 0.9983162571623986]
    x2 = [0.09308574095423705, 0.9525867434178398, 0.9936801742865224, 0.15146531531710972, 0.4554124588600381, 0.6446704949372436, 0.8207023743393098, 0.36281802630862336, 0.4994854756965742, 0.9691733737329793]
    x3 = [0.8304252632405683, 0.1559358972739967, 0.9929042563138192, 0.9610077550214366, 0.9479530178839192, 0.41124769279667195, 0.15671394738414102, 0.10414233652225635, 0.3122747970589639, 0.9584110356074136]
    x4 = [0.006715663211737555, 0.8632222082963186, 0.4381912107003699, 0.9948406368871704, 0.8341155724693066, 0.4133997962187874, 0.5979890722809864, 0.005796810844911793, 0.15534264062091596, 0.20211826785802967]
    x5 = [0.3594501812077451, 0.5153239839617291, 0.2537221500704121, 0.20457916800334086, 0.215056956924785, 0.07640398715123564, 0.15087693482799058, 0.8303866677715736, 0.5491071782674977, 0.8560052218606929]
    x6 = [0.32905894624038545, 0.8316121681139131, 0.7152333047370752, 0.28630169021836915, 0.5648006715733932, 0.4023718198402364, 0.5297108761293481, 0.8355393273409718, 0.9170256719863896, 0.8752207629826034]
    x7 = [0.04060032589777429, 0.624933130475544, 0.8558427668996439, 0.23432979757469719, 0.6836116067502549, 0.21487040696193282, 0.7013809903722572, 0.15375829086896997, 0.5000588456150364, 0.2074004090971463]
    x40000 = [1.0, 0.39785356208747785, 0.41449017573646474, 0.6283871004008136, 1.0, 0.5507204092653762, 0.0954875646428419, 0.0, 0.4210614841605472, 0.6974056237968949]
    x40001 = [1.0, 0.6009824599371598, 0.8005257760087661, 0.5679259091500694, 0.7099141300124654, 0.30715234645180145, 0.0, 0.23424537565419945, 0.35375334370073286, 0.9096199555093505]
    x39995 = [0.3588175339486661, 0.5449028617264218, 1.0, 1.0, 0.0, 1.0, 0.0, 0.29441979776379984, 0.0, 1.0]
    x39963 = [0.7757237914914716, 0.5764882071993569, 0.2807163837736093, 0.19352791678387482, 1.0, 0.0, 0.38160135318508487, 0.0, 0.579437986623093, 1.0]
    x39907 = [0.4482183477205983, 0.027409116218383683, 0.25874372878562424, 0.6394764496282036, 1.0, 0.0, 1.0, 0.4692043394584843, 0.566542139621032, 0.0]


    xtry = [x1, x2, x3, x4, x5, x6, x7, x40000, x40001, x39995, x39963, x39907] #try on small subset of data
    xtry = np.array(xtry)

    #for xtry in xx:
    print('Ensemble')
    print(Ensemble.predict(xtry))
    print('RF')
    print(RF.predict(xtry))
    xtry = xgb.DMatrix(xtry)
    print('XGB6')
    print(XGB6.predict(xtry))
    print('XGB18')
    print(XGB18.predict(xtry))
    print('XGB60')
    print(XGB60.predict(xtry))
    print('XGB240')
    print(XGB240.predict(xtry))
else:
    print("Loading data...")
    folder_path = r"C:\Users\20205209\OneDrive - TU Eindhoven\TUE\Code\BO Summer School Hasselt\BO windwake lab\Raw data of the EXPensive Optimization benchmark library (EXPObench)_2_all\Windwake_1000iter_10runs"
    iter_df = load_data(folder_path)
    #x_train, y_train, x_val, y_val = get_data_all(iter_df)
    x_train, y_train = get_data_all(iter_df)
    print("Done loading")
    xtry = x_train #try on all data

    print('Ensemble MAE')
    print(mean_absolute_error(y_train, Ensemble.predict(xtry)))
    print('RF MAE')
    print(mean_absolute_error(y_train, RF.predict(xtry)))
    xtry = xgb.DMatrix(xtry)
    print('XGB6 MAE')
    print(mean_absolute_error(y_train, XGB6.predict(xtry)))
    print('XGB18 MAE')
    print(mean_absolute_error(y_train, XGB18.predict(xtry)))
    print('XGB60 MAE')
    print(mean_absolute_error(y_train, XGB60.predict(xtry)))
    print('XGB240 MAE')
    print(mean_absolute_error(y_train, XGB240.predict(xtry)))





