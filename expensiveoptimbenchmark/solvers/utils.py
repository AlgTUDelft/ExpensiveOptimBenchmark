import time

class Monitor:

    def __init__(self):
        self.time_before_eval = []
        self.time_after_eval = []
        self.fitness_after_eval = []

    def start(self):
        self.time_after_eval = [time.time()]
        self.time_after_eval = []

    def end(self):
        self.time_before_eval.append(time.time())

    def commit_start_eval(self):
        self.time_before_eval.append(time.time())

    def commit_end_eval(self, r):
        self.time_after_eval.append(time.time())
        self.fitness_after_eval.append(r)

    def eval_time(self):
        return [t_after_eval_prev - t_before_eval for (t_before_eval, t_after_eval_prev) in zip(self.time_before_eval, self.time_after_eval)]

    def model_time(self):
        return [t_after_eval - t_before_eval for (t_before_eval, t_after_eval) in zip(self.time_before_eval, self.time_after_eval[1:])]