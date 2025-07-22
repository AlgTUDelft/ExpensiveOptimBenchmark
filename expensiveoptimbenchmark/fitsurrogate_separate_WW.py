import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns; sns.set_style(style='white') #sns.set()
from sklearn.metrics import mean_squared_error, max_error, mean_absolute_error
from sklearn import svm
import time
import pickle


def get_exp_ids(folder_path):
    '''
    Get all exp_id values, sorted by approach

    Parameters
    ----------
    folder_path : path
        Folder path.

    Returns
    -------
    exp_id_all : dataframe
        Dataframe with exp_id values separated into approaches (columns) and runs (rows).

    '''
    # Loads all iter.csv files in the folder
    log_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    file_list = [f for f in log_files if f.endswith('iters.csv')]
    # Read log files
    iter_df = pd.DataFrame()
    for f in file_list:
        temp = pd.read_csv(f)
        iter_df = pd.concat([iter_df,temp])
    
    
    # Extract data and separate into approaches
    list_solver = np.unique(iter_df['approach'].values)
    exp_id_all = pd.DataFrame()
    for sol in list_solver:
        tempsol = iter_df.loc[iter_df['approach']==sol]
        for expid in tempsol:
            exp_id_all[sol] = np.unique(tempsol['exp_id'].values)
            
    return exp_id_all



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
        
def get_train_data(approach, run, iters, iter_df, exp_id_all):
    '''
    Get training set of data for a single run and a single approach.

    Parameters
    ----------
    approach : string
        Approach/method, for example DONEjl.
    run : int
        Which run to look at. Assumes all approaches have the same nr of runs.
    iters: int
        Number of iterations to use in training data.

    Returns
    -------
    x_train : numpy array
        Training data for inputs.
    y_train : numpy array
        Training labels.

    '''

            
            
    #print(iter_df['iter_fitness'].isnull().values.any())
    
    
    # Make dataset for each approach and run separately
    df = iter_df.loc[iter_df['exp_id']==exp_id_all[approach].iloc[run]]
    df = df.loc[df['iter_idx']<iters]
    y_train = df['iter_fitness']
    y_train = y_train.to_numpy(dtype=float)
    y_train = y_train*1e-9 # for windwake problem, to convert from Wh to GWh
    x_train = df['iter_x']
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
    print('Train Data labels:', y_train.shape)
    print('Train Data inputs:', x_train.shape)
    return x_train, y_train

def get_test_data(iter_df,test_size):
    '''
    Get training set of data for a single run and a single approach.

    Parameters
    ----------
    iter_df : dataframe
    size : int
        Total number of test samples.

    Returns
    -------
    x_test : numpy array
        Test data for inputs.
    y_test : numpy array
        Test labels.

    '''
    
    # Sort by fitness value
    df = iter_df.sort_values(by='iter_fitness')
    df = df.iloc[0:test_size]
    
    y_test = df['iter_fitness']
    y_test = y_test.to_numpy(dtype=float)
    y_test = y_test*1e-9 # for windwake problem, to convert from Wh to GWh
    x_test = df['iter_x']
    x_test = x_test.to_numpy()
    
    temp = np.array([])
    for x in x_test:
        #x = list(map(str.strip, x.strip('][').replace('"', '').split(',')))
        x = x.strip('[]')
        x = np.fromstring(x,dtype=float,sep=',')
        if temp.size==0:
            temp = np.copy(x)
        else:
            temp = np.vstack((temp,x))
    #x_all = x_all.to_numpy(dtype=float)
    #x_all= np.stack(x_all,axis=0)
    x_test = np.copy(temp)
    print('Test Data labels:', y_test.shape)
    print('Test Data inputs:', x_test.shape)
    return x_test, y_test

# Train model
def train(folder_path, approach, run,iters):
    #Load data
    print("Loading data...")
    iter_df = load_data(folder_path)
    exp_id_all = get_exp_ids(folder_path)
    x_train, y_train = get_train_data(approach,run,iters,iter_df,exp_id_all)
    x_val, y_val = get_test_data(iter_df,1000)
    #x_all, y_all = load_data(folder_path)
    print("Done loading")
    num_train = x_train.shape[0]
    if num_train != y_train.shape[0]:
        print('Warning: different number of input and output samples for training set.')
    num_val = x_val.shape[0]
    if num_val != y_val.shape[0]:
        print('Warning: different number of input and output samples for validation set.')
    
    # #Randomize data rows
    # np.random.seed(1234) 
    # np.random.shuffle(x_all)
    # np.random.shuffle(y_all)
    
    # #Split data
    # x_train = x_all[0:int(np.ceil(0.8*num_samples)),:] #80% training
    # y_train = y_all[0:int(np.ceil(0.8*num_samples))] #80% training
    # x_val = x_all[int(np.ceil(0.8*num_samples)):int(np.ceil(0.9*num_samples)),:] #10% validation
    # y_val = y_all[int(np.ceil(0.8*num_samples)):int(np.ceil(0.9*num_samples))] #10% validation
    # x_test = x_all[int(np.ceil(0.9*num_samples)):,:] #10% validation
    # y_test = y_all[int(np.ceil(0.9*num_samples)):] #10% validation    
    
    #print('x_train', x_train)
    #print('y_train', y_train)
    #print('x_val', x_val)
    #print('y_val', y_val)
    #print('x_test', x_test)
    #print('y_test', y_test)
    
        
    # Train MVRSM model
    import sys
    #sys.path.append("C:/Users/20205209/surfdrive/TUE/Code/ExpensiveBenchmarkSurvey/benchmarkingsurvey/OfflineSurrogates")
    trainMVRSM = 0
    if trainMVRSM:
        import MVRSM_fitsurrogate as mvrsm
        #MVRSM_fitsurrogate 
        print('Training MVRSM regressor...')
        problem = 'WW'
        if problem=='ESP':
            d = 49
            num_int = 49
            lb = 0*np.ones(d,dtype=float)
            ub = 7*np.ones(d,dtype=float)
        elif problem=='PD':
            d = 10
            num_int=0
            lb = np.copy([-0.01, -0.05, -0.01, -0.05, -0.01, -0.05, -0.01, -0.05, -0.01, -0.05])
            ub = np.copy([0.287397, 0.014, 0.287397, 0.014, 0.287397, 0.014 , 0.287397, 0.014, 0.287397, 0.014])
        elif problem=='WW':
            d = 10
            num_int = 10
            lb = 0*np.ones(d,dtype=float)
            ub = 1*np.ones(d,dtype=float)
        t0 = time.time()
        MVRSM = mvrsm.train(x_train,y_train,num_int, lb, ub)
        t1 = time.time()
        MVRSM_time = t1-t0

        # Save surrogate
        MVRSMfile = "MVRSM.pkl"
        with open(MVRSMfile, 'wb') as file:
            pickle.dump(MVRSM, file)
        
        MVRSM_train = mvrsm.predict(x_train,MVRSM)
        MVRSM_val = mvrsm.predict(x_val,MVRSM)
        MVRSM_trainmse = mean_squared_error(y_train, MVRSM_train)
        MVRSM_trainmax = max_error(y_train, MVRSM_train)
        MVRSM_trainmae = mean_absolute_error(y_train, MVRSM_train)
        MVRSM_valmse = mean_squared_error(y_val, MVRSM_val)
        MVRSM_valmax = max_error(y_val, MVRSM_val)
        MVRSM_valmae = mean_absolute_error(y_val, MVRSM_val)

        
        print('Training time: ', MVRSM_time)
        print('Train MSE: ', MVRSM_trainmse)
        print('Validation MSE: ', MVRSM_valmse)
        print('Train MAE: ', MVRSM_trainmae)
        print('Validation MAE: ', MVRSM_valmae)
        print('Train max: ', MVRSM_trainmax)
        print('Validation max: ', MVRSM_valmax)
        
    
    
    #Train linear model
    
    trainLM = 0
    if trainLM:
        from sklearn import linear_model
        LM = linear_model.LinearRegression()
        print('Training linear regressor...')
        t0 = time.time()
        LM.fit(x_train,y_train)
        t1 = time.time()
        LM_time = t1-t0
        print('Done training linear regressor...')
        LM_train = LM.predict(x_train)
        LM_val = LM.predict(x_val)
        LM_trainmse = mean_squared_error(y_train, LM_train)
        LM_trainmax = max_error(y_train, LM_train)
        LM_trainmae = mean_absolute_error(y_train, LM_train)
        LM_valmse = mean_squared_error(y_val, LM_val)
        LM_valmax = max_error(y_val, LM_val)
        LM_valmae = mean_absolute_error(y_val, LM_val)
        print('Training time: ', LM_time)
        print('Train MSE: ', LM_trainmse)
        print('Validation MSE: ', LM_valmse)
        print('Train MAE: ', LM_trainmae)
        print('Validation MAE: ', LM_valmae)
        print('Train max: ', LM_trainmax)
        print('Validation max: ', LM_valmax)
        
        
        # Save surrogate
        LMfile = "LM.pkl"
        with open(LMfile, 'wb') as file:
            pickle.dump(LM, file)
    
    
    #Train a random forest
    
    trainRF = 1
    if trainRF:
        from sklearn.ensemble import RandomForestRegressor
        simpleRF = RandomForestRegressor()  
        print('Training random forest regressor...')
        t0 = time.time()
        simpleRF.fit(x_train,y_train)
        t1 = time.time()
        simpleRF_time = t1-t0
        print('Done training random forest regressor...')
        simpleRF_train = simpleRF.predict(x_train)
        simpleRF_val = simpleRF.predict(x_val)
        simpleRF_trainmse = mean_squared_error(y_train, simpleRF_train)
        simpleRF_trainmax = max_error(y_train, simpleRF_train)
        simpleRF_trainmae = mean_absolute_error(y_train, simpleRF_train)
        simpleRF_valmse = mean_squared_error(y_val, simpleRF_val)
        simpleRF_valmax = max_error(y_val, simpleRF_val)
        simpleRF_valmae = mean_absolute_error(y_val, simpleRF_val)
        
        print('Training time: ', simpleRF_time)
        print('Train MSE: ', simpleRF_trainmse)
        print('Validation MSE: ', simpleRF_valmse)
        print('Train MAE: ', simpleRF_trainmae)
        print('Validation MAE: ', simpleRF_valmae)
        print('Train max: ', simpleRF_trainmax)
        print('Validation max: ', simpleRF_valmax)
        
        # Save surrogate
        RFfile = "RF.pkl"
        with open(RFfile, 'wb') as file:
            pickle.dump(simpleRF, file)
    
    
    #Train XGBoost
    
    trainXGB=1
    if trainXGB:
        import xgboost as xgb
    
        dtrain = xgb.DMatrix(x_train, label=y_train)
        dval = xgb.DMatrix(x_val, label=y_val)
        
        #params = {'max_depth': 6, 'eta': 0.3, 'objective': 'reg:squarederror'}
        #num_round=2
        params = {}
        
        # # Load surrogate
        # XGBfile = "XGB.pkl"
        # with open(XGBfile, 'rb') as file:
        #     XGB = pickle.load(file)
        
        print('Training XGBoost regressor...')
        t0 = time.time()
        XGB = xgb.train(params,dtrain)
        t1 = time.time()
        XGB_time = t1-t0
        print('Done training XGBoostregressor...')
        XGB_train = XGB.predict(dtrain)
        XGB_val = XGB.predict(dval)
        XGB_trainmse = mean_squared_error(y_train, XGB_train)
        XGB_trainmax = max_error(y_train, XGB_train)
        XGB_trainmae = mean_absolute_error(y_train, XGB_train)
        XGB_valmse = mean_squared_error(y_val, XGB_val)
        XGB_valmax = max_error(y_val, XGB_val)
        XGB_valmae = mean_absolute_error(y_val, XGB_val)

        
        print('Training time: ', XGB_time)
        print('Train MSE: ', XGB_trainmse)
        print('Validation MSE: ', XGB_valmse)
        print('Train MAE: ', XGB_trainmae)
        print('Validation MAE: ', XGB_valmae)
        print('Train max: ', XGB_trainmax)
        print('Validation max: ', XGB_valmax)
        
        # Save surrogate
        XGBfile = "XGB.pkl"
        with open(XGBfile, 'wb') as file:
            pickle.dump(XGB, file)
        
    #Train MLP
    trainMLP = 0
    if trainMLP:
        from sklearn.neural_network import MLPRegressor
        MLP = MLPRegressor(max_iter=500)  
        print('Training MLP regressor...')
        t0 = time.time()
        MLP.fit(x_train,y_train)
        t1 = time.time()
        MLP_time = t1-t0
        print('Done training MLP regressor...')

        # Save surrogate
        MLPfile = "MLP.pkl"
        with open(MLPfile, 'wb') as file:
            pickle.dump(MLP, file)

        MLP_train = MLP.predict(x_train)
        MLP_val = MLP.predict(x_val)
       
        
        MLP_trainmse = mean_squared_error(y_train, MLP_train)
        MLP_trainmax = max_error(y_train, MLP_train)
        MLP_trainmae = mean_absolute_error(y_train, MLP_train)
        MLP_valmse = mean_squared_error(y_val, MLP_val)
        MLP_valmax = max_error(y_val, MLP_val)
        MLP_valmae = mean_absolute_error(y_val, MLP_val)

        
        print('Training time: ', MLP_time)
        print('Train MSE: ', MLP_trainmse)
        print('Validation MSE: ', MLP_valmse)
        print('Train MAE: ', MLP_trainmae)
        print('Validation MAE: ', MLP_valmae)
        print('Train max: ', MLP_trainmax)
        print('Validation max: ', MLP_valmax)
    
    #Train GP
    trainGP = 1
    if trainGP:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern
        kernel = 1.0 * Matern(length_scale=1.0, nu=2.5)
        GP = GaussianProcessRegressor(kernel=kernel,alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5)
        print('Training Gaussian Process regressor...')
        t0 = time.time()
        GP.fit(x_train,y_train)
        t1 = time.time()
        GP_time = t1-t0
        print('Done training Gaussian Process regressor...')

        # Save surrogate
        GPfile = "GP.pkl"
        with open(GPfile, 'wb') as file:
            pickle.dump(GP, file)

        GP_train = GP.predict(x_train)
        GP_val = GP.predict(x_val)        
        
        GP_trainmse = mean_squared_error(y_train, GP_train)
        GP_trainmax = max_error(y_train, GP_train)
        GP_trainmae = mean_absolute_error(y_train, GP_train)
        GP_valmse = mean_squared_error(y_val, GP_val)
        GP_valmax = max_error(y_val, GP_val)
        GP_valmae = mean_absolute_error(y_val, GP_val)

        
        print('Training time: ', GP_time)
        print('Train MSE: ', GP_trainmse)
        print('Validation MSE: ', GP_valmse)
        print('Train MAE: ', GP_trainmae)
        print('Validation MAE: ', GP_valmae)
        print('Train max: ', GP_trainmax)
        print('Validation max: ', GP_valmax)

    
    # Train random Fourier model
    import sys
    #sys.path.append("C:/Users/20205209/surfdrive/TUE/Code/ExpensiveBenchmarkSurvey/benchmarkingsurvey/OfflineSurrogates")
    trainRFE = 0
    if trainRFE:
        import DONE_fitsurrogate as rfe
        #MVRSM_fitsurrogate 
        print('Training random Fourier regressor...')

        t0 = time.time()
        RFE = rfe.train(x_train,y_train)
        t1 = time.time()
        RFE_time = t1-t0

        # Save surrogate
        RFEfile = "RFE.pkl"
        with open(RFEfile, 'wb') as file:
            pickle.dump(RFE, file)
        
        RFE_train = rfe.predict(x_train,RFE)
        RFE_val = rfe.predict(x_val,RFE)
        RFE_trainmse = mean_squared_error(y_train, RFE_train)
        RFE_trainmax = max_error(y_train, RFE_train)
        RFE_trainmae = mean_absolute_error(y_train, RFE_train)
        RFE_valmse = mean_squared_error(y_val, RFE_val)
        RFE_valmax = max_error(y_val, RFE_val)
        RFE_valmae = mean_absolute_error(y_val, RFE_val)

        
        print('Training time: ', RFE_time)
        print('Train MSE: ', RFE_trainmse)
        print('Validation MSE: ', RFE_valmse)
        print('Train MAE: ', RFE_trainmae)
        print('Validation MAE: ', RFE_valmae)
        print('Train max: ', RFE_trainmax)
        print('Validation max: ', RFE_valmax)    


# Return nice labels
def convert_feature_label(y_feature):
    if y_feature == 'iter_best_fitness':
        return 'Best fitness value'
    else:
        return y_feature

# Plot iter log files from a folder path
def plot_iter_file(folder_path, y_feature = 'iter_best_fitness', save_file=None):

    # Get iter log files from folder path 
    log_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    file_list = [f for f in log_files if f.endswith('iters.csv')]

    # Initialise figure
    fig = plt.figure(figsize=(9/1.5,4/1.5))
    ax = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    #plt.subplots_adjust(left=0.14, bottom=0.16, right=0.96, top=0.90, wspace=0.41, hspace=0.2)
    plt.subplots_adjust(top=0.93,bottom=0.185,left=0.105,right=0.985,hspace=0.2,wspace=0.39)
    #plt.style.use('seaborn-colorblind')
    #colours = ['tab:gray', 'tab:green', 'tab:red', 'tab:purple', 'tab:blue', 'tab:pink', 'tab:olive']
    colours = ['y','g','c','r','m','b', 'tab:green']
    sns.set_color_codes("colorblind")
    iter_df = pd.DataFrame()
    
    # Read log files
    for f in file_list:
        temp = pd.read_csv(f)
        iter_df = pd.concat([iter_df,temp])
        #iter_df = pd.read_csv(f)
    #solver = iter_df['approach'].values[0]
    #problem = iter_df['problem'].values[0]
    list_solver = np.unique(iter_df['approach'].values)
    #list_problem = np.unique(iter_df['problem'].values)
    
    iter_df = iter_df.astype({'exp_id':float})
    
    list_exp_id = np.unique(iter_df['exp_id'])

    
    
    solvernamedict = {
        'CoCaBO': 'CoCaBO',
        'MVRSM/advanced': 'MVRSM',
        'hyperopt/randomsearch': 'RS',
        'hyperopt/randomsearch/conditional': 'RS',
        'hyperopt/tpe': 'HO',
        'hyperopt/tpe/conditional': 'HO',
        'smac': 'SMAC-nondet',
        'smac/det': 'SMAC',
        'smac/det/ac': 'SMAC',
        'smac/ac': 'SMAC',
        'bayesianoptimization': 'BO',
        'DONEjl': 'DONE',
        'IDONE/advanced/normal': 'IDONE'
        }
    
    solvermarkerdict = {
        'CoCaBO': '^',
        'MVRSM/advanced': '*',
        'hyperopt/randomsearch': 'o',
        'hyperopt/randomsearch/conditional': 'o',
        'hyperopt/tpe': 's',
        'hyperopt/tpe/conditional': 's',
        'smac': 'd',
        'smac/det': 'd',
        'smac/det/ac': 'd',
        'smac/ac': 'd',
        'bayesianoptimization': 'v',
        'DONEjl': '.',
        'IDONE/advanced/normal': 'p'
        }
    
    solverorderdict = {
        'CoCaBO': 4,
        'MVRSM/advanced': 5,
        'hyperopt/randomsearch': 0,
        'hyperopt/randomsearch/conditional': 0,
        'hyperopt/tpe': 1,
        'hyperopt/tpe/conditional': 1,
        'smac': 2,
        'smac/det': 2,
        'smac/det/ac': 2,
        'smac/ac': 2,
        'bayesianoptimization': 3,
        'DONEjl': 4,
        'IDONE/advanced/normal': 6
        }
    
    
    #list_solver = [list_solver[i] for i in solverorder] #reorder list
    list_solver = sorted(list_solver, key=solverorderdict.get)
    if 'smac' in list_solver:
        list_solver.remove('smac') #only use SMAC deterministic case
        print('smac removed')
    if 'smac/ac' in list_solver:
        list_solver.remove('smac/ac') #only use SMAC deterministic case
        print('smac/ac removed')
    
    corrvec_obj = [] # Vector used to calculate correlation (objective)
    corrvec_time = [] # Vector used to calculate correlation (time to propose candidate solution)
    for solver in list_solver:
        
        fitness_value_df = pd.DataFrame()
        time_value_df = pd.DataFrame()
        for exp_id in list_exp_id:
            exp_df = iter_df[iter_df['exp_id'] == exp_id]
            exp_df =  exp_df[exp_df['approach'] == solver]
            exp_df = exp_df.reset_index()
            fitness_value_df = pd.concat([fitness_value_df, exp_df[y_feature]], axis=1)
            time_value_df = pd.concat([time_value_df, exp_df['iter_model_time']], axis=1)

        try:
            fitness_value_df = fitness_value_df.apply(lambda x: x.str.strip("[]"))
            fitness_value_df = fitness_value_df.astype(float)
        except:
            pass     
        random_eval = 49 #Number initial random evaluations: 300 for HPO, 49 for ESP, 20 for windwake, 20 for pitzdaily
        skip_random_evaluations = 1 #whether to show random evaluations in the plot (0) or not (1)
        # if skip_random_evaluations:
        #     fitness_value_df = fitness_value_df[random_eval:] #skip random evaluations
        #     fitness_value_df = fitness_value_df.reset_index(drop=True)
        #     time_value_df = time_value_df[random_eval:] #skip random evaluations
        #     time_value_df = time_value_df.reset_index(drop=True)
        #For student t test
        test = fitness_value_df.iloc[[-1]].values.tolist()
        #print(solver,test)
        print(solver,[x for x in test[0] if str(x)!='nan'])
        print((~np.isnan(test)).sum(1))
        #
        
        # For six-hump camel, show distance to optimum
        if iter_df['problem'].values[0] == "Six-hump camel(d_int=0, d_cont=2, log=False)":
            fitness_value_df['iter_best_fitness']=fitness_value_df['iter_best_fitness']+1.0316
        
        
        # Save for spearman/kendall correlation test
        temp = fitness_value_df.values
        temp = temp[:,~np.all(np.isnan(temp), axis=0)]
        save_all_iters = 0 #if you want to save all iterations
        
        if save_all_iters==1:
            for x in temp:
                for y in x:
                    corrvec_obj.append(y)
        else:
            for y in temp[-1]:
                corrvec_obj.append(y)
        #print('hi',corrvec, len(corrvec))
        
        temp = time_value_df.values
        temp = temp[:,~np.all(np.isnan(temp), axis=0)]
        temp = np.cumsum(temp,axis=0)
        if save_all_iters==1:
            for x in temp:
                for y in x:
                    corrvec_time.append(y)
        else:
            for y in temp[-1]:
                corrvec_time.append(y)
        #print(len(corrvec_obj),len(corrvec_time))
        
        
        normalize=1 #1 for true, 0 for false
        
        mean_fitness_value = fitness_value_df.mean(axis=1)
        sd_fitness_value = fitness_value_df.std(axis=1)
        if "randomsearch" in solver:
            randommean = mean_fitness_value
        if normalize:
            mean_fitness_value = (mean_fitness_value-randommean[random_eval-1])/(randommean[random_eval-1]-randommean[0])+1
            sd_fitness_value = sd_fitness_value/(randommean[random_eval-1]-randommean[0])
        
        if skip_random_evaluations:
            mean_fitness_value = mean_fitness_value[random_eval:] #skip random evaluations
            mean_fitness_value = mean_fitness_value.reset_index(drop=True)
            sd_fitness_value = sd_fitness_value[random_eval:] #skip random evaluations
            sd_fitness_value = sd_fitness_value.reset_index(drop=True)
            time_value_df = time_value_df[random_eval:] #skip random evaluations
            time_value_df = time_value_df.reset_index(drop=True)
        upperError = pd.to_numeric(mean_fitness_value + sd_fitness_value)
        lowerError = pd.to_numeric(mean_fitness_value - sd_fitness_value)
        mean_time_value = time_value_df.mean(axis=1)
        sd_time_value = time_value_df.std(axis=1)
        upperError_time = pd.to_numeric(mean_time_value + sd_time_value)
        lowerError_time = pd.to_numeric(mean_time_value - sd_time_value)
        
        
        markevery = int(len(upperError)/10)
        ax.plot(mean_fitness_value.index, mean_fitness_value, label = solvernamedict[solver], linewidth=2, markevery=markevery,marker=solvermarkerdict[solver], color=colours[solverorderdict[solver]])
        ax.fill_between(pd.to_numeric(mean_fitness_value.index), upperError, lowerError, alpha=0.25, color=colours[solverorderdict[solver]])
        #plt.xlim(np.min(fitness_value_df.index), np.max(fitness_value_df.index)+1)
        
        ax2.plot(mean_time_value.index, mean_time_value, label = solvernamedict[solver], linewidth=0.5, markevery=markevery,marker=solvermarkerdict[solver], color=colours[solverorderdict[solver]]) #linewidth=0.5 for big plots
        ax2.fill_between(pd.to_numeric(mean_time_value.index), upperError_time, lowerError_time, alpha=0.25, color=colours[solverorderdict[solver]])
        #plt.xlim(np.min(time_value_df.index), np.max(time_value_df.index)+1)
        
        
        
    #plt.style.use('seaborn-colorblind')   
    
    
    
    
    ax.grid(True)
    ax2.grid(True)
    leg = ax.legend(ncol=2, mode='expand') #for ESP
    #leg = ax.legend() #for HPO
    #leg.set_draggable(True)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Normalized objective")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Computation time [s]")
    #leg = ax2.legend()
    #leg.set_draggable(True)
    ax2.set_yscale('log')
    ymin2, ymax2 = ax2.get_ylim()
    #ax2.set_ylim([ymin2,ymax2*5000])
    ax1log = 0 #1 for Ackley53 and six hump camel
    if ax1log:
        ax.set_yscale('log')
        ymin1, ymax1 = ax.get_ylim()
        ax.set_ylim([ymin1,ymax1]) #*2000 with legend
    else:
        ymin1, ymax1 = ax.get_ylim()
        ax.set_ylim([ymin1,(ymax1-ymin1)*1.5+ymin1]) #1.7 for plots with legend
        #if normalize:
            #ax.set_ylim([1,2])
        
    #ax.set_xlim([-5,100]) #for linearmivabo
    #ax2.set_xlim([-5,100]) #for linearmivabo

    print(f"{solver}:\nmean={mean_fitness_value.values[-1]} sd={sd_fitness_value.values[-1]}")
    # Styling
    #plt.title(f"Problem {problem}")
    #plt.ylabel(convert_feature_label(y_feature))
    





    # Spearman/kendall correlation between time spent to propose a candidate solution and objective
    fig2 = plt.figure()
    plt.scatter(corrvec_time,corrvec_obj)
    Spearman = spearmanr(corrvec_time,corrvec_obj)
    #print(Spearman)
    Kendall = kendalltau(corrvec_time,corrvec_obj)
    print(Kendall)

    # print(spearmanr([1,2,3,5,6],[3,6,9,15,18]))
    # print(kendalltau([1,2,3,5,6],[3,6,9,15,18]))
    # print(spearmanr([1,2,3,5,6],[-0.5,0.2,-0.33,0.3,0.0]))
    # print(kendalltau([1,2,3,5,6],[-0.5,0.2,-0.33,0.3,0.0]))





    # Display
    

    if save_file is not None:
        fig.savefig(save_file)
    else:
        plt.show()

if __name__ == "__main__":
    import sys
    folder_path = sys.argv[1]
    if len(sys.argv) > 2:
        feature = sys.argv[2]
    else:
        feature = 'iter_best_fitness'
    #plot_iter_file(folder_path, feature)
    
    #methods (ESP): IDONE/advanced/normal, DONEjl, smac/det/ac, MVRSM/advanced, bayesianoptimization, hyperopt/tpe, hyperopt/randomsearch
    #methods (PD): same without IDONE, with smac/det
    #methods (WW): same as PD
    
    #methods = ['IDONE/advanced/normal', 'DONEjl', 'smac/det/ac', 'MVRSM/advanced', 'bayesianoptimization', 'hyperopt/tpe', 'hyperopt/randomsearch'] #ESP
    methods = ['DONEjl', 'smac/det', 'MVRSM/advanced', 'bayesianoptimization', 'hyperopt/tpe', 'hyperopt/randomsearch'] #PD/WW
    
    for method in methods:
        print('*******\n*******\nNow switching to method', method, '*******\n*******\n')
        for i in range(10):
            print('******\nStarting training on method', method, 'run', i, '.\n******')
            train(folder_path, method, i, 500)
            print('******\nFinished training on method', method, 'run', i, '.\n******')
    
    