import numpy as np
from typing import Union

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

## Utilities
def maybe_int(v: Union[int, None]):
    if v is None:
        return None
    else:
        return int(v)

def maybe_float(v: Union[float, None]):
    if v is None:
        return None
    else:
        return float(v)