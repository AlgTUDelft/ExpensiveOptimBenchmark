import subprocess

class DockerCFDBenchmarkProblem:

    def __init__(self, name, d, lbs, ubs, vartype, direction, errval):
        self.name = name
        self.d = d
        self.lb = lbs
        self.ub = ubs
        self.vt = vartype
        self.direction = 1 if direction == "min" else -1
        self.errval = errval
    
    def evaluate(self, xs):
        evalCommand = f"docker run --rm frehbach/cfd-test-problem-suite ./dockerCall.sh {self.name}"
        parsedCandidate = ",".join(["%.8f" % x for x in xs])
        cmd = f"{evalCommand} '{parsedCandidate}'"
        # print(f"Running '{cmd}'")
        # res = subprocess.check_output(cmd, shell=True)
        res = subprocess.check_output(["docker", "run", "--rm", "frehbach/cfd-test-problem-suite", "./dockerCall.sh", self.name, parsedCandidate])
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

# TODO: Verify that the ESP problem is indeed a minimization problem.
ESP = DockerCFDBenchmarkProblem("ESP", 49, [0] * 49, [7] * 49, ['int'] * 49, "min", 1.0)

PitzDaily = DockerCFDBenchmarkProblem("PitzDaily",
    10,
    [-0.01, -0.05, -0.01, -0.05, -0.01, -0.05, -0.01, -0.05, -0.01, -0.05],
    [0.287397, 0.014, 0.287397, 0.014, 0.287397, 0.014 , 0.287397, 0.014, 0.287397, 0.014],
    ['cont'] * 10, "min", 1.0)

KaplanDuct = DockerCFDBenchmarkProblem("KaplanDuct", 4, 
    [720.0, -1050.5, 726.99988, -710.0],
    [3609.0, -390.0, 3609.0,  450.0], 
    ['cont'] * 4, "max", 1.0)

dockersimbenches = [ESP, PitzDaily, KaplanDuct]
# Note: HeatExchanger is a 28 dimensional problem, all continuous, unlike previous problems.
# lb: [ -1, -1, -1, -1,  0,  0,  0,  0, 0, 0, -1, -1,  0,  0,  0,  0, 0, 0, -1, -1,  0,  0,  0,  0, 0, 0, -1, -1]
# ub: [  1,  1,  1,  1, 10, 10, 10, 10, 1, 1,  1,  1, 10, 10, 10, 10, 1, 1,  1,  1, 10, 10, 10, 10, 1, 1,  1,  1]