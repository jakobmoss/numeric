# cython: language_level=3

# General modules
import numpy as np
import math
import auxA as mcint
import sys

#
# Test-functions to integrate
#
def f1(q):
    """
    y * cos(x)
    """
    return q[1] * math.cos(q[0])   


#
# Main function
#
def mainA():
    """
    Test of the Monte Carlo routines
    """
    ##############
    # Function 1 #
    ##############
    # Initializations
    a1 = np.array([-math.pi/2.0, 0])
    b1 = np.array([math.pi/2.0, 1])
    exact1 = 1
    N1 = 1000

    # Pretty print!
    print('\n ** Integrating y*cos(x) from (x,y) = (-pi/2,0) to (pi/2,1) ** ')
    print('Exact solution = {0:d}'.format(exact1))

    # Perform the integration
    res1, err1 = mcint.plainmc(f1, a1, b1, N1)
    print('- Plain Monte Carlo integration:')
    print('Sampling (N):  = {0:d}'.format(N1))
    print('Integral       = {0:.15f}'.format(res1))
    print('Error estimate = {0:.5e}'.format(err1))
    print('Actual error   = {0:.5e}'.format(abs(res1 -  exact1)))
