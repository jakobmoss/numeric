# cython: language_level=3

# General modules
import numpy as np
import math
import sys

#
# Plain Monte Carlo integration
#
def plainmc(F, a, b, N):
    """
    
    Arguments:
    - `F`: Function to integrate
    - `a`: Vector of stating point
    - `b`: Vector of end point
    - `N`: Number of points to sample
    """
    # Initilizations
    volume = 1
    x      = np.zeros(a.shape, dtype='float')
    s      = 0  # Sum
    ss     = 0  # Sum of Squares
    
    # Calculate volume
    for i in range(len(a)):
        volume *= b[i] - a[i]

    # Sample the N points
    for i in range(N):
        x = np.random.uniform(a, b, len(a))  # SEE NOTE 1
        y = F(x)
        s += y
        ss += y*y

    # Calculate average and sigma=sqrt(variance)
    mean  = s/N
    sigma = math.sqrt(ss / N - mean*mean)

    # Return estimate of result and error
    err = volume * sigma/math.sqrt(N)
    res = volume * mean
    return res, err


# NOTE 1
# This should be equivalent to
#  >  x = a + np.random.uniform(0, 1, len(a)) * (b - a)
# which is just a vectorized version of the function in the lecture notes.
