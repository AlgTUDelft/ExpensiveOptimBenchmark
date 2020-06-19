import random
import time

def csvify(value):
    if not isinstance(value, str):
        v = str(value)
    else:
        v = value
    if  '"' in v or  ',' in v or '\n' in v:
        v.replace("\"", "\"\"")
        return f"\"{v}\""
    else:
        return v

class Monitor:

    def __init__(self, solver, problem, log):
        self.solver = solver
        self.problem = problem
        self.expuid = f"{time.time()}{random.randint(0, 1<<14)}".replace(".", "")
        self.time_before_eval = []
        self.time_after_eval = []
        self.iter_fitness = []
        self.iter_x = []
        self.iter_best_fitness = []
        self.iter_best_x = []
        self.best_fitness = None
        self.best_x = None
        self.num_iters = 0
        self.log = log

    def start(self):
        self.time_after_eval = [time.time()]
        self.time_before_eval = []

    def end(self):
        self.time_before_eval.append(time.time())

    def commit_start_eval(self):
        self.time_before_eval.append(time.time())

    def commit_end_eval(self, x, r):
        self.time_after_eval.append(time.time())
        self.iter_fitness.append(r)
        xc = list(x).copy()
        self.iter_x.append(xc)
        if self.best_fitness is None or r < self.best_fitness:
            self.best_fitness = r
            self.best_x = xc
        self.iter_best_fitness.append(self.best_fitness)
        self.iter_best_x.append(self.best_x)
        self.num_iters += 1

        # Write logs every `write_every` iterations
        if self.log['write_every'] is not None and self.num_iters % self.log['write_every'] == 0:
            from_iter = self.num_iters - self.log['write_every']
            # to_iter = None is fine as we want to write out everything so far.
            with open(self.log['file_iters'], 'a') as f:
                self.emit_csv_iters(f, from_iter=from_iter, emit_header=self.log['emit_header'])
            self.log['emit_header'] = False

    def eval_time_gen(self, from_iter=0, to_iter=None):
        # Note: first after eval is start.
        return (a - b for (b, a) in zip(self.time_before_eval[from_iter:to_iter], self.time_after_eval[(from_iter+1):to_iter]))
        
    def eval_time(self, from_iter=0, to_iter=None):
        return list(self.eval_time_gen(from_iter=from_iter, to_iter=to_iter))

    def model_time_gen(self, from_iter=0, to_iter=None):
        return (a - b for (b, a) in zip(self.time_after_eval[from_iter:to_iter], self.time_before_eval[from_iter:to_iter]))

    def model_time(self, from_iter=0, to_iter=None):
        return list(self.model_time_gen(from_iter=from_iter, to_iter=to_iter))

    def total(self):
        return self.time_before_eval[-1] - self.time_after_eval[1]

    def emit_csv_iters(self, file, emit_header=None, from_iter=0, to_iter=None):
        # Emit header if starting from the first iteration.
        # Otherwise, assume we are appending to a pre-existing file
        # and no new header should be added in.
        if emit_header is None:
            emit_header = from_iter == 0
        
        if emit_header:
            file.write("approach,problem,exp_id,iter_idx,iter_eval_time,iter_model_time,iter_fitness,iter_x,iter_best_fitness,iter_best_x\n")

        to_iter_num = self.num_iters if to_iter is None else to_iter

        for iter_idx, iter_eval_time, iter_model_time, iter_fitness, iter_x, iter_best_fitness, iter_best_x in \
            zip(range(from_iter, to_iter_num),
                self.eval_time_gen(from_iter=from_iter, to_iter=to_iter),
                self.model_time_gen(from_iter=from_iter, to_iter=to_iter),
                self.iter_fitness[from_iter:to_iter],
                self.iter_x[from_iter:to_iter],
                self.iter_best_fitness[from_iter:to_iter],
                self.iter_best_x[from_iter:to_iter]):
            file.write(f"{csvify(self.solver)},{csvify(self.problem)},{self.expuid},{iter_idx},{iter_eval_time},{iter_model_time},{iter_fitness},{csvify(iter_x)},{iter_best_fitness},{csvify(iter_best_x)}\n")

    def emit_csv_summary(self, file, emit_header=True):
        if emit_header:
            file.write("approach,problem,exp_id,total_iters,total_time,total_model_time,total_eval_time,best_fitness,best_x\n")
        total_time = self.time_before_eval[-1] - self.time_after_eval[1]
        total_model_time = sum(self.model_time_gen())
        total_eval_time = sum(self.eval_time_gen())
        file.write(f"{csvify(self.solver)},{csvify(self.problem)},{self.expuid},{self.num_iters},{total_time},{total_model_time},{total_eval_time},{self.best_fitness},{csvify(self.best_x)}\n")

import numpy as np
class Binarizer:
    def __init__(self, mask, lb, ub):
        self.din = len(lb)
        self.lb = lb
        self.ub = ub
        self.in_mask = mask
        # One-to-one mapping to begin with.
        vars_each = np.ones(self.din, np.int)
        self.shift = np.zeros(self.din)
        self.shift[mask] = -lb[mask]
        # Binarization requires ceil(log2(range + 1)) new variables. 
        vars_each[mask] = np.ceil(np.log2(ub[mask] - lb[mask] + 1))
        self.dout = sum(vars_each)
        self.out_mask = np.asarray([False] * self.dout)
        # Construct binarization weight matrices
        self.W = np.zeros((self.din, self.dout))
        self.Winv = np.zeros((self.dout, self.din))
        self.m = np.ones(self.dout) * np.inf
        # Bounds for binarized vector.
        self.blb = np.zeros((self.dout))
        self.bub = np.zeros((self.dout))
        cvars_each = np.cumsum(vars_each)
        for i_in in range(self.din):
            start = cvars_each[i_in - 1] if i_in != 0 else 0
            end = cvars_each[i_in]
            for idx, i_out in enumerate(range(start, end)):
                self.W[i_in, i_out] = 2.0 ** -idx
                self.Winv[i_out, i_in] = 2.0 ** idx
                self.m[i_out] = 2.0 if vars_each[i_in] != 1 else np.inf
                self.out_mask[i_out] = self.in_mask[i_in]
                self.blb[i_out] = 0.0 if vars_each[i_in] != 1 else self.lb[i_in]
                self.bub[i_out] = 1.0 if vars_each[i_in] != 1 else self.ub[i_in]

    def binarize(self, x):
        xv = np.matmul(x + self.shift, self.W)
        xv[self.out_mask] = np.floor(xv[self.out_mask])
        return xv % self.m

    def unbinarize(self, x):
        return np.clip(np.matmul(x, self.Winv) - self.shift, self.lb, self.ub)

    def ubs(self):
        return self.bub

    def lbs(self):
        return self.blb