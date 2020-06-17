import subprocess

class DockerCFDBenchmarkProblem:

    def __init__(self, name, d, lbs, ubs, vartype, direction):
        self.name = name
        self.d = d
        self.lb = lbs
        self.ub = ubs
        self.vt = vartype
        self.direction = 1 if direction == "min" else -1
    
    def evaluate(self, xs):
        evalCommand = f"docker run --rm frehbach/cfd-test-problem-suite ./dockerCall.sh {self.name}"
        parsedCandidate = ",".join([str(x) for x in xs])
        return self.direction * float(subprocess.check_output(f"{evalCommand} '{parsedCandidate}'", shell=True).strip().split(b"\n")[-1])

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
ESP = DockerCFDBenchmarkProblem("ESP", 49, [0] * 49, [7] * 49, ['int'] * 49, "min")

PitzDaily = DockerCFDBenchmarkProblem("PitzDaily",
    10,
    [-0.01, -0.05, -0.01, -0.05, -0.01, -0.05, -0.01, -0.05, -0.01, -0.05],
    [0.287397, 0.014, 0.287397, 0.014, 0.287397, 0.014 , 0.287397, 0.014, 0.287397, 0.014],
    ['cont'] * 10, "min")

KaplanDuct = DockerCFDBenchmarkProblem("KaplanDuct", 4, 
    [720.0, -1050.5, 726.99988, -710.0],
    [3609.0, -390.0, 3609.0,  450.0], 
    ['cont'] * 4, "max")

dockersimbenches = [ESP, PitzDaily, KaplanDuct]
# Note: HeatExchanger is a 28 dimensional problem, all continuous, unlike previous problems.
# lb: [ -1, -1, -1, -1,  0,  0,  0,  0, 0, 0, -1, -1,  0,  0,  0,  0, 0, 0, -1, -1,  0,  0,  0,  0, 0, 0, -1, -1]
# ub: [  1,  1,  1,  1, 10, 10, 10, 10, 1, 1,  1,  1, 10, 10, 10, 10, 1, 1,  1,  1, 10, 10, 10, 10, 1, 1,  1,  1]