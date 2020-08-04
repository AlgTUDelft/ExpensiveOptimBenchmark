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
        return None
