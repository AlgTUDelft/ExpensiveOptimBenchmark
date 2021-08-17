## This problem requires docker, namely the docker file mrebolle/r-geccoc:Track1.
## This file can be found at https://hub.docker.com/r/mrebolle/r-geccoc
## For more information about this problem, see https://www.th-koeln.de/informatik-und-ingenieurwissenschaften/gecco-2021-industrial-challenge-call-for-participation_82086.php

from .base import BaseProblem

import subprocess
import platform
import numpy as np
import os

class DockerHospitalBenchmarkProblem(BaseProblem):

    def __init__(self, name, d, lbs, ubs, vartype, direction, errval):
        self.name = name
        self.d = d
        self.lb = np.asarray(lbs)
        self.ub = np.asarray(ubs)
        self.vt = np.asarray(vartype)
        self.direction = 1 if direction == "min" else -1
        self.errval = errval

        if os.path.exists("./evaluate.sh"):
            self.evalCommand = ["./evaluate.sh"]
        elif os.path.exists("/evaluate.sh"):
            self.evalCommand = ["/evaluate.sh"]
        else:
            # Note: experiment should be ran with sudo under linux to allow this command to work,
            # or under an user in the docker group.
            self.evalCommand = ["docker", "run", "--rm", "mrebolle/r-geccoc:Track1", "-c"]
    
    def evaluate(self, xs):
        concatenatedCandidate = ",".join(["%.8f" % x if xvt == 'cont' else "%i" % x for (x, xvt) in zip(xs, self.vt)])
        parsedCandidate = f"Rscript objfun.R \"{concatenatedCandidate}\""
        cmd = f"{self.evalCommand + [parsedCandidate]}"
        print(f"Running '{cmd}'")
        # res = subprocess.check_output(cmd, shell=True)
        res = subprocess.check_output(self.evalCommand + [parsedCandidate])
        reslast = res.strip().split(b"\n")[-1]
        print(f"Result: {res}. Objective: {reslast}")
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
        return f"Hospital()"

Hospital = DockerHospitalBenchmarkProblem("hospital", 29, 
    np.array([ 6,  7, 3, 3, 3, 5, 3, 3, 25, 17, 2, 1, 0.25, 0.05, 0.07, 0.005, 0.07, 1e-04, 0.08, 0.25, 0.08, 0.5, 1e-6, 2, 1e-6, 1e-6, 1, 2,  0.5 ]), 
    np.array([14, 13, 7, 9, 7, 9, 5, 7, 35, 25, 5, 7, 2   , 0.15, 0.11, 0.02 , 0.13, 0.002, 0.12, 0.35, 0.12, 0.9, 0.01, 4, 1.1, 0.0625, 2, 5, 0.75]), 
    ['cont'] * 29, "min", 10000.0)

dockerhospitalsimbenches = [Hospital]