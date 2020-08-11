
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
    idone_Sa = 5 
    idone_Ar = 6 

    # Ablation analysis
    TS_idone = 10
    binarize_idone = 11
    uniform_idone = 12 
    larger_expl_idone = 13 
    TS_uniform_idone = 14
    binarize_larger_idone = 15



# Controls the strategies
IDONE_LIST = [Strategies.idone, Strategies.idone_Sa, Strategies.idone_Ar,
              Strategies.TS_idone, Strategies.binarize_idone, Strategies.uniform_idone, Strategies.larger_expl_idone,
              Strategies.TS_uniform_idone, Strategies.binarize_larger_idone
             ]
THOMPSON_SAMPLING = [Strategies.idone_Sa, Strategies.TS_idone, Strategies.TS_uniform_idone]
BINARIZE = [Strategies.idone_Sa, Strategies.binarize_idone, Strategies.binarize_larger_idone]
UNIFORM_SAMPLING = [Strategies.idone_Ar, Strategies.uniform_idone, Strategies.TS_uniform_idone]
LARGER_PROBABILITY = [Strategies.idone_Ar, Strategies.larger_expl_idone, Strategies.binarize_larger_idone]
SCALING = []

class Problems(enum.Enum):
    maxcut = 0
    rosen = 1
    tsp = 2
    esp = 3


def get_solver(strategy):
    if strategy in IDONE_LIST:
        return 'idone'
    elif strategy in [Strategies.smac]:
        return 'smac'
    elif strategy in [Strategies.hyperopt]:
        return 'hyperopt'
    elif strategy in [Strategies.randomsearch]:
        return 'randomsearch'
    elif strategy in [Strategies.bayesianoptimization]:
        return 'bayesianoptimization'
    else:
        raise ValueError("Strategy does not exist.")

# Wrapper function for scripting run_experiment.py
def run_exp(problem_enum, strategy_enum, args, out_path):

    command = [
            "python",
            "run_experiment.py",
            f"--max-eval={args.evaluations}", 
            f"--out-path={out_path}", 
            f"--repetitions={args.repetitions}", 
            problem_enum.name
            ]

    # Problem parameters            
    if problem_enum != Problems.esp and problem_enum != Problems.tsp:
        command.append(f'-d={args.dimensions}')
    
    if problem_enum == Problems.tsp:
        instance_file_path = f'TSP_instances/ftv{args.dimensions}.atsp'
        if not os.path.exists(instance_file_path):
            raise ValueError("Invalid dimension given, instance does not exist.")
        command.append(f'--tsplib-file={instance_file_path}')
    if args.binarize is True:
        command.append('--binarize=true')
    if args.seed is not None and problem_enum == Problems.maxcut:
        command.append(f'--graph-seed={args.seed}')

    # Solver and parameters
    command.append(get_solver(strategy_enum))
    if strategy_enum != Strategies.randomsearch:
        #command.append(f'--rand-evals={args.randevals}') TODO
        pass
    # Solver parameters
    if strategy_enum in THOMPSON_SAMPLING:
        command.append('--sampling=thompson')
    if strategy_enum in BINARIZE: 
        command.append('--binarize-categorical=true')
        command.append('--binarize-int=true')
    if strategy_enum in SCALING:
        command.append('--scaling=true')
    if strategy_enum in UNIFORM_SAMPLING:
        command.append('--sampling=uniform')
    if strategy_enum in LARGER_PROBABILITY:
        command.append('--expl-prob=larger')
        
        
    print("\nRunning command:\n", str(command))
    subprocess.run(command)

    # Document command 
    with open(f"{out_path}command_log.txt", 'a') as f:
        f.write(" ".join(command))
        f.write("\n")


if __name__ == "__main__":
    
    # Parse arguments
    parser = ArgumentParser()

    parser.add_argument('-p', '--problem', type=str, nargs = '+', required=True,
                        help='Pick problem', choices=[p.name for p in Problems])
    parser.add_argument('-s', '--solver', type=str, nargs = '+', required=True,
                        help='Pick solver', choices=[s.name for s in Strategies])


    parser.add_argument('-d','--dimensions', type=int, default = 25,
                        help='Dimension of decision variables to the problem')
    parser.add_argument('-e','--evaluations', type=int, default=100,
                        help='Number of evaluations for the given problem instance')
    parser.add_argument('-r','--repetitions', type=int, default=1,
                        help='Number of experiments to perform')
    parser.add_argument('-re', '--randevals', type=int, default=1,
                        help='Number of random evaluations')
    parser.add_argument('-t', '--tag', type=str,
                        help='Optional tag id for experiment')
    parser.add_argument('-b', '--binarize', action='store_true',
                        help='Use binarized problem version if available')
    parser.add_argument('--seed', type=int, default=None,
                            help="Define random seed if problem allows this as input")


    args = parser.parse_args()


    # Cast strings to enum class
    strategies = [Strategies[s] for s in args.solver]
    problem_instance_list = [Problems[s] for s in args.problem]

    for problem_instance_enum in problem_instance_list:
    
        experiment_id = f"{datetime.now().strftime('%d%m_%H%M%S')}_{problem_instance_enum.name}{'_'+args.tag if args.tag is not None else ''}"
        out_path = f"./results/{experiment_id}/"

        # Run experiments
        for s in strategies:
            run_exp(problem_instance_enum, s, args, out_path)

        # Plot results (saves image in log folder)
        plot_iter_file(out_path, save_file=f"{out_path}/iter_plot.png")


            