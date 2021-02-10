import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns; sns.set_style(style='white') #sns.set()


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
    colours = ['y','g','c','r','m','b']
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
    list_problem = np.unique(iter_df['problem'].values)
    
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
        'bayesianoptimization': 'BO'
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
        'bayesianoptimization': 'v'
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
        'bayesianoptimization': 3
        }
    
    
    #list_solver = [list_solver[i] for i in solverorder] #reorder list
    list_solver = sorted(list_solver, key=solverorderdict.get)
    if 'smac' in list_solver:
        list_solver.remove('smac') #only use SMAC deterministic case
        print('smac removed')
    if 'smac/ac' in list_solver:
        list_solver.remove('smac/ac') #only use SMAC deterministic case
        print('smac/ac removed')
    
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
        random_eval = 24
        fitness_value_df = fitness_value_df[random_eval:] #skip random evaluations
        fitness_value_df = fitness_value_df.reset_index(drop=True)
        time_value_df = time_value_df[random_eval:] #skip random evaluations
        time_value_df = time_value_df.reset_index(drop=True)
        #For student t test
        test = fitness_value_df.iloc[[-1]].values.tolist()
        #print(solver,test)
        print(solver,[x for x in test[0] if str(x)!='nan'])
        print((~np.isnan(test)).sum(1))
        #
        print(list_problem[0])
        mean_fitness_value = fitness_value_df.mean(axis=1)
        if "cvxnonsep_psig20" in list_problem[0]:
            mean_fitness_value -= 93.81138788
        if "cvxnonsep_psig30" in list_problem[0]:
            mean_fitness_value -= 78.99885434
        if "cvxnonsep_psig40" in list_problem[0]:
            mean_fitness_value -= 85.49576764
        sd_fitness_value = fitness_value_df.std(axis=1)
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
    ax.set_ylabel("Objective")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Computation time [s]")
    #leg = ax2.legend()
    #leg.set_draggable(True)
    ax2.set_yscale('log')
    ymin2, ymax2 = ax2.get_ylim()
    #ax2.set_ylim([ymin2,ymax2*5000])
    ax1log = 0 #1 for Ackley53
    if "cvxnonsep_psig20" in list_problem[0] or "cvxnonsep_psig30" in list_problem[0] or "cvxnonsep_psig40" in list_problem[0]:
        ax1log = 1 #Also use log scale for cvxnonsep problem
        ax.set_ylabel("Distance from optimum")
    if ax1log:
        ax.set_yscale('log')
        ymin1, ymax1 = ax.get_ylim()
        ax.set_ylim([ymin1,ymax1*100]) #*2000 with legend
    else:
        ymin1, ymax1 = ax.get_ylim()
        ax.set_ylim([ymin1,(ymax1-ymin1)*1.5+ymin1]) #1.7 for plots with legend
        
    #ax.set_xlim([-5,100]) #for linearmivabo
    #ax2.set_xlim([-5,100]) #for linearmivabo

    print(f"{solver}:\nmean={mean_fitness_value.values[-1]} sd={sd_fitness_value.values[-1]}")
    # Styling
    #plt.title(f"Problem {problem}")
    #plt.ylabel(convert_feature_label(y_feature))
    

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
    plot_iter_file(folder_path, feature)