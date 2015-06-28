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
    return math.cos(q[0]) * math.sin(q[1])


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
    b1 = np.array([math.pi/2.0, math.pi])
    exact1 = 4
    N1 = 1000

    # Pretty print!
    print('\n ** Integrating cos(x)*sin(y) from (x,y) = (-pi/2,0) to (pi/2,pi) ** ')
    print('Exact solution = {0:d}'.format(exact1))

    # Perform the integration
    res1, err1 = mcint.plainmc(f1, a1, b1, N1)
    print('- Plain Monte Carlo integration:')
    print('Sampling (N):  = {0:d}'.format(N1))
    print('Integral       = {0:.15f}'.format(res1))
    print('Error estimate = {0:.5e}'.format(err1))
    print('Actual error   = {0:.5e}'.format(abs(res1 -  exact1)))
    print('\nA plot of how the error scales with N is shown in plot.A.pdf',
          '(the data is kept in Aerr.dat)')

    # Error-estimation as a function of N
    #  .. it is almost abusive to use stderr for this..
    print('# Error-estimate from part A', file=sys.stderr)
    for k in range(1, 6):
        N = int(math.pow(10, k))
        res, err = mcint.plainmc(f1, a1, b1, N)
        print('{0:12d} {1:16.12f}'.format(N, err), file=sys.stderr)
