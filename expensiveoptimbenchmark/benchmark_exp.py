
import subprocess
import os
from datetime import datetime
import enum
from argparse import ArgumentParser
from plot_utils import plot_iter_file

class Strategies(enum.Enum):
    idone = 0
    smac = 1
    hyperopt = 2
    randomsearch = 3
    bayesianoptimization = 4
    idone_Sa = 5 #TODO: This strategy are not implemented
    idone_Ar = 6

class Problems(enum.Enum):
    maxcut = 0
    rosen = 1
    tsp = 2
    esp = 3


def get_solver(strategy):
    if strategy in [Strategies.idone, Strategies.idone_Sa, Strategies.idone_Ar]:
        return 'idone'
    elif strategy in [Strategies.smac]:
        return 'smac'
    elif strategy in [Strategies.hyperopt]:
        return 'hyperopt'
    elif strategy in [Strategies.randomsearch]:
        return 'randomsearch'
    elif strategy in [Strategies.bayesianoptimsation]:
        return 'bayesianoptimsation'
    else:
        raise ValueError("Strategy does not exist.")

# Wrapper function for scripting run_experiment.py
def run_exp(problem, strategy, args, out_path):

    command = [
            "python",
            "run_experiment.py",
            f"--max-eval={args.evaluations}", 
            f"--out-path={out_path}", 
            f"--repetitions={args.repetitions}", 
            problem, f'-d={args.dimensions}',
            get_solver(strategy)
            ]
    
    # IDONE parameters
    TS = strategy in [Strategies.idone_Sa]
    binarize = strategy in [Strategies.idone_Sa]
    if TS is True:
        command.append('--thompson-sampling=true')
    if binarize is True: 
        command.append('--binarize-categorical=true')
        command.append('--binarize-int=true')


    print("Running", str(command))
    subprocess.run(command)

    # Document command 
    with open(f"{out_path}command_log.txt", 'w+') as f:
        f.write(" ".join(command))
        f.write("\n")


if __name__ == "__main__":
    
    # Parse arguments
    parser = ArgumentParser()

    parser.add_argument('-p', '--problem', type=str, nargs = '+',
                        help='Pick problem', choices=[p.name for p in Problems])
    parser.add_argument('-s', '--solver', type=str, nargs = '+',
                        help='Pick solver', choices=[s.name for s in Strategies])

    parser.add_argument('-d','--dimensions', type=int, default = 25,
                        help='Dimension of decision variables to the problem')
    parser.add_argument('-e','--evaluations', type=int, default=100,
                        help='Number of evaluations for the given problem instance')
    parser.add_argument('-r','--repetitions', type=int, default=1,
                        help='Number of experiments to perform')


    args = parser.parse_args()

    # Parameters
    problem_instance_list = args.problem

    # Cast solver strings to Strategies enum class
    strategies = [Strategies[s] for s in args.solver]

    for problem_instance in problem_instance_list:
        
        experiment_id = f"{datetime.now().strftime('%d%m_%H%M%S')}_{problem_instance}"
        out_path = f"./results/{experiment_id}/"

        # Run experiments
        for s in strategies:
            run_exp(problem_instance, s, args, out_path)

        # Plot results (saves image in log folder)
        plot_iter_file(out_path, save_file=f"{out_path}/iter_plot.png")


            