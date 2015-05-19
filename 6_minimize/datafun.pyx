# cython: language_level=3

import numpy as np
import math

#
# Rosenbrock
#
def rosen(q):
    """
    The Rosenbrock valley function: f(x,y) = (1 - x)^2 - 100*(y - x^2)^2
    """
    x = q[0]
    y = q[1]
    return math.pow(1-x, 2) + 100*math.pow(y-x*x, 2)


def grad_rosen(q):
    """
    The first derivatives of the function above in the form of the gradient
    """
    # Init
    df = np.zeros(2, dtype='float64')
    x = q[0]
    y = q[1]

    # Calculate and return
    df[0] = 2*(x-1) - 400*(y-x*x)*x 
    df[1] = 200*(y-x*x)
    return df


def hes_rosen(q):
    """
    Second derivatives of the function above in the form of the Hessian matrix
    """
    # Init
    H = np.zeros((2, 2), dtype='float64')
    x = q[0]
    y = q[1]

    # Fill the Hessian and return it
    H[0, 0] = 2 - 400*y + 1200*x*x
    H[0, 1] = -400*x
    H[1, 0] = -400*x
    H[1, 1] = 200
    return H


#
# Himmelblau
#
def himmel(q):
    """
    The Himmelblau function: f(x,y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2
    """
    x = q[0]
    y = q[1]
    return math.pow(x*x+y-11, 2) + math.pow(x+y*y-7, 2)


def grad_himmel(q):
    """
    The first derivatives of the function above in the form of the gradient
    """
    # Init
    df = np.zeros(2, dtype='float64')
    x = q[0]
    y = q[1]

    # Calculate and return
    df[0] = 4*(x*x+y-11)*x + 2*(x+y*y-7)
    df[1] = 2*(x*x+y-11) + 4*(x+y*y-7)*y
    return df


def hes_himmel(q):
    """
    Second derivatives of the function above in the form of the Hessian matrix
    """
    # Init
    H = np.zeros((2, 2), dtype='float64')
    x = q[0]
    y = q[1]

    # Fill the Hessian and return it
    H[0, 0] = 12*x*x + 4*y - 42
    H[0, 1] = 4*x + 4*y
    H[1, 0] = 4*x + 4*y
    H[1, 1] = 4*x + 12*y*y - 26
    return H
