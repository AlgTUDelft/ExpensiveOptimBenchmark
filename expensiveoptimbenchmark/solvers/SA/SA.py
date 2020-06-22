import numpy as np
import math

def SA_minimize(f, x0, lb, ub, N = 1000, perc = None, Tf = None, T0 = None, seed=None):
    d = len(x0)
    # Defaults for temperature.
    perc = 1 / d if perc is None else perc
    T0 = -1/math.log(0.8) if T0 is None else T0
    Tf = ((1e-8)/T0)**(1/N) if Tf is None else Tf
    
    # Storage for solutions
    Xeval = np.zeros((d, N))
    Yeval = np.zeros((1, N))
    Xbest = np.zeros((d, N))
    Ybest = np.zeros((1, N))

    Xeval[:, 0] = x0
    Yeval[0, 0] = f(x0)
    Xbest[:, 0] = x0
    Ybest[0, 0] = Yeval[0, 0]

    rng = np.random.RandomState(seed=seed)
    current_X = x0.copy()

    T = T0

    for i in range(1, N):
        # Modify every dimension with probability perc
        altered = rng.uniform(size=d) > perc
        # Avoid going past boundaries.
        is_upperbounded = current_X == ub
        is_lowerbounded = current_X == lb
        # No modifications possible if the bounds are equal!
        altered[np.logical_and(is_upperbounded, is_lowerbounded)] = False
        offset_upper = np.choose(is_upperbounded, [1, -1])
        offset_lower = np.choose(is_lowerbounded, [-1, 1])
        # Draw a modification vector, and add it to next_X.
        next_X = current_X + np.choose(altered * rng.randint(1, 2+1, size=d), [0, offset_upper, offset_lower])
        
        # Evaluate
        Xeval[:, i] = next_X
        Yeval[0, i] = f(next_X)

        # r = "unk"
        if Yeval[0, i] < Ybest[0, i - 1]:
            current_X = next_X
            Xbest[:, i] = next_X
            Ybest[0, i] = Yeval[0, i]
            # r = "Improved!"
        elif rng.uniform() < math.exp((Ybest[0, i - 1] - Yeval[0, i]) / T): 
            current_X = next_X
            Xbest[:, i] = next_X
            Ybest[0, i] = Yeval[0, i]
            # r = "Accepting worse."
        else:
            Xbest[:, i] = Xbest[:, i - 1]
            Ybest[0, i] = Ybest[0, i - 1]
            # r = "Failure."

        # Update temperature.
        T = T * Tf
        # print(f"Completed iteration. {r}. Temperature: {T}. Current X: {current_X}, best y: {Ybest[0, i]}")

    # Get best result from history.

    opt_idx = np.argmin(Yeval)
    xopt = Xeval[:, opt_idx]
    yopt = Yeval[0, opt_idx]

    return yopt, xopt