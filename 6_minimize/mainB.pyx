# cython: language_level=3

import numpy as np
import auxB as minimize
import math, sys

# Global variables
import globvar

# Different functions to root-find
from datafun import *


#
# Main function
#
def mainB():
    """
    Test of the minimization methods
    """
    ## PART ZERO  --> Pretty print and general init!
    print('\n**************************************************')
    print('* Test of Quasi-Newton\'s method for minimization *')
    print('**************************************************\n')

    # General settings for tolerence
    alpha = 1e-4
    eps = 1e-9

    # ##
    # ## PART ONE  -->  Minimization of the Rosenbrock valley function by running
    # ##                Newton's method with user supplied derivatives
    # print('\n -- Minimization of the Rosenbrock valley function --')
    # print('\nFunction: f(x,y) = (1 - x)^2 - 100*(y - x^2)^2')

    # # Initialization
    # x0 = np.array([-1, 2], dtype='float64')
    # print('\nStarting point:\nx0 =', x0, '\nf(x0) =', rosen(x0))

    # # Run the root search with numerical derivatives
    # globvar.steps = 0  # Redundant, but more clear this way
    # mini = minimize.newton_min(rosen, grad_rosen, hes_rosen, x0, alpha, eps)
    # print('\nSolution:\nx =', mini, '\nf(x) =', rosen(mini),
    #       '\nf\'(x)', grad_rosen(mini))
    # print('Steps used: ', globvar.steps)


    # ##
    # ## PART TWO  -->  Minimization of the Himmelblau function by running
    # ##                Newton's method with user supplied derivatives
    # print('\n -- Minimization of the Himmelblau function --')
    # print('\nFunction: f(x,y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2')

    # # Initialization
    # x0 = np.array([2, 5], dtype='float64')
    # print('\nStarting point:\nx0 =', x0, '\nf(x0) =', himmel(x0))

    # # Run the root search with numerical derivatives
    # globvar.steps = 0  # Redundant, but more clear this way
    # mini = minimize.newton_min(himmel, grad_himmel, hes_himmel, x0, alpha, eps)
    # print('\nSolution:\nx =', mini, '\nf(x) =', himmel(mini),
    #       '\nf\'(x)', grad_himmel(mini))
    # print('Steps used: ', globvar.steps)
