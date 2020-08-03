import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import markers
import os

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
    fig = plt.figure()
    ax = fig.add_subplot()

    # To read experiments with the same solver from multiple files, use a dict
    dict_dataframes = {} # solver name as key, dataframe as value

    # Read log files
    iter_dfs = pd.concat(pd.read_csv(f) for f in file_list)

    for (_, iter_df) in iter_dfs.groupby(['approach', 'problem']):
        solver = iter_df['approach'].values[0]
        problem = iter_df['problem'].values[0]
        list_exp_id = np.unique(iter_df['exp_id'].values)

        if solver in dict_dataframes:
            fitness_value_df = dict_dataframes.get(solver)

        else:
            fitness_value_df = pd.DataFrame()


        for exp_id in list_exp_id:
            exp_df = iter_df[iter_df['exp_id'] == exp_id]
            exp_df = exp_df.reset_index()
            fitness_value_df = pd.concat([fitness_value_df, exp_df[y_feature]], axis=1)
    
        try:
            fitness_value_df = fitness_value_df.apply(lambda x: x.str.strip("[]"))
            fitness_value_df = fitness_value_df.astype(float)
        except:
            pass 

        dict_dataframes[solver] = fitness_value_df

    # Read and plot from dataframes dict
    for solver, fitness_value_df in dict_dataframes.items():

        mean_fitness_value = fitness_value_df.mean(axis=1)
        sd_fitness_value = fitness_value_df.std(axis=1)
        upperError = pd.to_numeric(mean_fitness_value + sd_fitness_value)
        lowerError = pd.to_numeric(mean_fitness_value - sd_fitness_value)
        
        ax.plot(mean_fitness_value.index, mean_fitness_value, label = solver, linewidth=2)
        ax.fill_between(pd.to_numeric(mean_fitness_value.index), upperError, lowerError, alpha=0.4)

        print(f"{solver}:\nmean={mean_fitness_value.values[-1]} sd={sd_fitness_value.values[-1]}")
    
    # Styling
    plt.title(f"Problem {problem}")
    plt.ylabel(convert_feature_label(y_feature))
    plt.xlabel("Iteration index")
    plt.xlim(mean_fitness_value.index[0], mean_fitness_value.index[-1])


    # Give all lines different markers
    valid_markers = ([item[0] for item in markers.MarkerStyle.markers.items() if item[1] is not 'nothing' and not item[1].startswith('tick') and not item[1].startswith('caret')])
    for i, line in enumerate(ax.get_lines()):
        line.set_marker(valid_markers[i])
        line.set_markevery(5)

    plt.style.use('seaborn-colorblind')
    plt.grid(True)

    # Display
    plt.legend()

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