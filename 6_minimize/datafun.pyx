# cython: language_level=3

import numpy as np
import math

#
# Rosenbrock
#
def rosen(q):
    """
    The Rosenbrock valley function (or actually the gradient of it)
    """
    # Init
    z = np.zeros(2, dtype='float64')
    x = q[0]
    y = q[1]

    # Calculate and return
    z[0] = 2*(x-1) - 400*(y-x*x)*x 
    z[1] = 200*(y-x*x)
    return z


def grad_rosen(q):
    """
    Derivatives of the function above in the form of the Jacobian
    """
    # Init
    J = np.zeros((2, 2), dtype='float64')
    x = q[0]
    y = q[1]

    # Fill the Jacobian and return it
    J[0, 0] = 2 - 400*y + 1200*x*x
    J[0, 1] = -400*x
    J[1, 0] = -400*x
    J[1, 1] = 200
    return J


#
# Himmelblau
#
def himmel(q):
    """
    The Himmelblau function (or actually the gradient of it)
    """
    # Init
    z = np.zeros(2, dtype='float64')
    x = q[0]
    y = q[1]

    # Calculate and return
    z[0] = 4*(x*x+y-11)*x + 2*(x+y*y-7)
    z[1] = 2*(x*x+y-11) + 4*(x+y*y-7)*y
    return z


def grad_himmel(q):
    """
    Derivatives of the function above in the form of the Jacobian
    """
    # Init
    J = np.zeros((2, 2), dtype='float64')
    x = q[0]
    y = q[1]

    # Fill the Jacobian and return it
    J[0, 0] = 12*x*x + 4*y - 42
    J[0, 1] = 4*x + 4*y
    J[1, 0] = 4*x + 4*y
    J[1, 1] = 4*x + 12*y*y - 26
    return J

