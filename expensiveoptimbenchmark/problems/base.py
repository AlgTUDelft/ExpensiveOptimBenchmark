import numpy as np

class BaseProblem:

    def __init__(self):
        pass

    def evaluate(self, x):
        raise NotImplementedError()

    def lbs(self):
        raise NotImplementedError()

    def ubs(self):
        raise NotImplementedError()

    def vartype(self):
        raise NotImplementedError()

    def dims(self):
        raise NotImplementedError()
    
    def dependencies(self):
        # Return a list of dims() Nones to signify no dependencies.
        return [None for _ in range(self.dims())]
