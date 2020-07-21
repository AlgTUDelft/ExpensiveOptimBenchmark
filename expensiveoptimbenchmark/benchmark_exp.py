
import subprocess
import os
import datetime
import time

from plot_utils import plot_iter_file

# Parameters
max_eval = 100
repetitions = 5
dimensions = 25
problem_instance = "rosen"
experiment_id = f"{datetime.date.today()}_{problem_instance}_{int(time.time())}"
out_path = f"./results/{experiment_id}/"

# Wrapper function for scripting run_experiment.py
def run_exp(problem, solver, args={}):

    print(f"\nRunning problem {problem} with solver {solver} and args={args}")
    
    command = [
            "python",
            "run_experiment.py",
            f"--max-eval={max_eval}", 
            f"--out-path={out_path}", 
            f"--repetitions={repetitions}", 
            problem, f'-d={dimensions}',
            solver
            ]
    
    TS = args['TS'] if 'TS' in args else False
    if solver == 'idone' and TS is True:
        command.append('--thompson-sampling=true')

    subprocess.run(command)


# Run experiments
run_exp(problem_instance, 'idone')
run_exp(problem_instance, 'idone', {'TS': True})
#run_exp(problem_instance, 'smac')
#run_exp(problem_instance, 'hyperopt')
#run_exp(problem_instance, 'randomsearch')
#run_exp(problem_instance, 'pygpgo')

# Plot results (saves image in log folder)
plot_iter_file(out_path, save_file=f"{out_path}/iter_plot.png")


            


