import random
import time

class Monitor:

    def __init__(self, solver, problem):
        self.solver = solver
        self.problem = problem
        self.expuid = f"{time.time()}{random.randint(0, 1<<14)}" 
        self.time_before_eval = []
        self.time_after_eval = []
        self.iter_fitness = []
        self.iter_best_fitness = []
        self.best_fitness = None

    def start(self):
        self.time_after_eval = [time.time()]
        self.time_before_eval = []

    def end(self):
        self.time_before_eval.append(time.time())

    def commit_start_eval(self):
        self.time_before_eval.append(time.time())

    def commit_end_eval(self, r):
        self.time_after_eval.append(time.time())
        self.iter_fitness.append(r)
        if self.best_fitness is None or r < self.best_fitness:
            self.best_fitness = r
        self.iter_best_fitness.append(self.best_fitness)

    def eval_time_gen(self, from_iter=0, to_iter=None):
        # Note: first after eval is start.
        return (a - b for (b, a) in zip(self.time_before_eval, self.time_after_eval[1:]))
        
    def eval_time(self, from_iter=0, to_iter=None):
        return list(self.eval_time_gen(from_iter=from_iter, to_iter=to_iter))

    def model_time_gen(self):
        return (a - b for (b, a) in zip(self.time_after_eval, self.time_before_eval))

    def model_time(self):
        return list(self.model_time_gen(from_iter=from_iter, to_iter=to_iter))

    def total(self):
        return self.time_before_eval[-1] - self.time_after_eval[1]

    def emit_csv(self, file, emit_header=None, from_iter=0, to_iter=None):
        # Emit header if starting from the first iteration.
        # Otherwise, assume we are appending to a pre-existing file
        # and no new header should be added in.
        if emit_header is None:
            emit_header = from_iter == 0
        
        if emit_header:
            file.write("approach,problem,iter_eval_time,iter_model_time,iter_fitness,iter_best_fitness\n")

        for iter_eval_time, iter_model_time, iter_fitness, iter_best_fitness in \
            zip(self.eval_time_gen(from_iter=from_iter, to_iter=to_iter),
                self.model_time_gen(from_iter=from_iter, to_iter=to_iter),
                self.iter_fitness[from_iter:to_iter],
                self.iter_best_fitness[from_iter:to_iter]):
            file.write(f"{self.solver},{self.problem},{iter_eval_time},{iter_model_time},{iter_fitness},{iter_best_fitness}\n")
