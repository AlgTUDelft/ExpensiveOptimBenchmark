from .base import BaseProblem

import subprocess
import numpy as np
import os

class DockerCFDBenchmarkProblem(BaseProblem):

    def __init__(self, name, d, lbs, ubs, vartype, direction, errval):
        self.name = name
        self.d = d
        self.lb = np.asarray(lbs)
        self.ub = np.asarray(ubs)
        self.vt = np.asarray(vartype)
        self.direction = 1 if direction == "min" else -1
        self.errval = errval
        if os.path.exists("./evaluate.sh"):
            self.evalCommand = ["./evaluate.sh", self.name]
        elif os.path.exists("/evaluate.sh"):
            self.evalCommand = ["/evaluate.sh", self.name]
        else:
            self.evalCommand = ["docker", "run", "--rm", "frehbach/cfd-test-problem-suite", "./dockerCall.sh", self.name]
    
    def evaluate(self, xs):
        parsedCandidate = ",".join(["%.8f" % x if xvt == 'cont' else "%i" % x for (x, xvt) in zip(xs, self.vt)])
        cmd = f"{self.evalCommand + [parsedCandidate]}"
        # print(f"Running '{cmd}'")
        # res = subprocess.check_output(cmd, shell=True)
        res = subprocess.check_output(self.evalCommand + [parsedCandidate])
        reslast = res.strip().split(b"\n")[-1]
        # print(f"Result: {res}")
        try:
            return self.direction * float(reslast)
        except:
            return self.errval

    def lbs(self):
        return self.lb

    def ubs(self):
        return self.ub

    def vartype(self):
        return self.vt

    def dims(self):
        return self.d

    def __str__(self):
        return f"DockerCFDBenchmark(name={self.name})"

ESP = DockerCFDBenchmarkProblem("ESP", 49, [0] * 49, [7] * 49, ['cat'] * 49, "min", 1.0)

ESP2 = DockerCFDBenchmarkProblem("ESP2", 49*2, [0, 28.0] * 49, [3, 146.0] * 49, ['cat', 'cont'] * 49, "min", 1.0)

ESP3 = DockerCFDBenchmarkProblem("ESP3", 5+49, [28.0] * 5 + [0] * 49, [146.0] * 5 + [3] * 49, ['cont'] * 5 + ['cat'] * 49, "min", 1.0)

PitzDaily = DockerCFDBenchmarkProblem("PitzDaily",
    10,
    [-0.01, -0.05, -0.01, -0.05, -0.01, -0.05, -0.01, -0.05, -0.01, -0.05],
    [0.287397, 0.014, 0.287397, 0.014, 0.287397, 0.014 , 0.287397, 0.014, 0.287397, 0.014],
    ['cont'] * 10, "min", 1.0)

KaplanDuct = DockerCFDBenchmarkProblem("KaplanDuct", 4, 
    [720.0, -1050.5, 726.99988, -710.0],
    [3609.0, -390.0, 3609.0,  450.0], 
    ['cont'] * 4, "max", 1.0)

dockersimbenches = [ESP, ESP2, ESP3, PitzDaily, KaplanDuct]
# Note: HeatExchanger is a 28 dimensional problem, all continuous, unlike previous problems.
# lb: [ -1, -1, -1, -1,  0,  0,  0,  0, 0, 0, -1, -1,  0,  0,  0,  0, 0, 0, -1, -1,  0,  0,  0,  0, 0, 0, -1, -1]
# ub: [  1,  1,  1,  1, 10, 10, 10, 10, 1, 1,  1,  1, 10, 10, 10, 10, 1, 1,  1,  1, 10, 10, 10, 10, 1, 1,  1,  1]