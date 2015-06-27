# cython: language_level=3

import numpy as np
import auxB as ode
import sys

# Global variables
import globvar


#
# ODEs to test the test the routines on
#
def f1(x, y):
    """
    ODE: y'' = -y   -->  cos, sin
    """
    globvar.calls += 1
    return np.array([y[1], -y[0]])

#
# Main function
#
def mainB():
    """
    Test of the ODE routines
    """
    # Initial setup
    acc  = 1e-3
    eps  = 1e-3

    ####################
    # Embedded stepper #
    ####################

    # Initial contitions
    a = 0
    b = 11
    yinit = np.array([0, 1], dtype='float')
    step = 0.1
    globvar.calls = 0

    # Evolve the system
    x23, y23, N23 = ode.rkdriver(f1, a, b, yinit, step, acc, eps, 'rk23')

    # Make output
    print('# Output from RK23: y0')
    for i in range(len(x23)):
        print(x23[i], y23[i, 0], N23[i], sep='\t')

    print('\n\n# Output from RK23: y1')
    for i in range(len(x23)):
        print(x23[i], y23[i, 1], N23[i], sep='\t')

        
    ####################
    # Runge's estimate #
    ####################
    # Initial contitions
    a = 0
    b = 11
    yinit = np.array([0, 1], dtype='float')
    step = 0.1
    globvar.calls = 0

    # Evolve the system
    x3, y3, N3 = ode.rkdriver(f1, a, b, yinit, step, acc, eps, 'rk3')

    # Make output
    print('\n\n# Output from RK3: y0')
    for i in range(len(x3)):
        print(x3[i], y3[i, 0], N3[i], sep='\t')

    print('\n\n# Output from RK3: y1')
    for i in range(len(x3)):
        print(x3[i], y3[i, 1], N3[i], sep='\t')


    #######################################################
    # Embedded stepper with order 2 and 1 -- Just for fun #
    #######################################################

    # Initial contitions
    a = 0
    b = 11
    yinit = np.array([0, 1], dtype='float')
    step = 0.1
    globvar.calls = 0

    # Evolve the system
    x12, y12, N12 = ode.rkdriver(f1, a, b, yinit, step, acc, eps, 'rk12')

    # Make output
    print('\n\n# Output from RK12: y0')
    for i in range(len(x12)):
        print(x12[i], y12[i, 0], N12[i], sep='\t')

    print('\n\n# Output from RK12: y1')
    for i in range(len(x12)):
        print(x12[i], y12[i, 1], N12[i], sep='\t')
