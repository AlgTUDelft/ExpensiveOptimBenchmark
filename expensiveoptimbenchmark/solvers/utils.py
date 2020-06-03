import random
import time

class Monitor:

    def __init__(self, solver, problem):
        self.solver = solver
        self.problem = problem
        self.expuid = f"{time.time()}{random.randint(0, 1<<14)}" 
        self.time_before_eval = []
        self.time_after_eval = []
        self.fitness_after_eval = []

    def start(self):
        self.time_after_eval = [time.time()]
        self.time_before_eval = []

    def end(self):
        self.time_before_eval.append(time.time())

    def commit_start_eval(self):
        self.time_before_eval.append(time.time())

    def commit_end_eval(self, r):
        self.time_after_eval.append(time.time())
        self.fitness_after_eval.append(r)

    def eval_time(self):
        # Note: first after eval is start.
        return [a - b for (b, a) in zip(self.time_before_eval, self.time_after_eval[1:])]

    def model_time(self):
        return [a - b for (b, a) in zip(self.time_after_eval, self.time_before_eval)]

    def total(self):
        return self.time_before_eval[-1] - self.time_after_eval[1]
