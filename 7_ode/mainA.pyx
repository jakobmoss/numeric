# cython: language_level=3

import numpy as np
import auxA as ode


#
# ODEs to test the test the routines on
#
def f1(x, y):
    """
    ODE: y'' = -y   -->  cos, sin
    """
    return np.array([y[1], -y[0]])

#
# Main function
#
def mainA():
    """
    Test of the ODE routines
    """
    # Initial setup
    acc  = 1e-3
    eps  = 1e-3
    step = 0.1

    # Initial contitions
    a = 0
    b = 8
    yinit = np.array([0, 1], dtype='float')

    # Evolve the system
    x1, y1 = ode.rkdriver(f1, a, b, yinit, step, acc, eps, 'rkstep23')
