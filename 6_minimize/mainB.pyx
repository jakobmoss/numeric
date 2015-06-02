# cython: language_level=3

import numpy as np
import auxB as minimize
import math, sys

# Global variables
import globvar

# Different functions to minimize
from datafun import *

# Import the root-finding routines from exercise 5
import roots as root


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

    ##
    ## PART ONE  -->  Minimization of the Rosenbrock valley function by running
    ##                Quasi Newton's method
    print('\n -- Minimization of the Rosenbrock valley function --')
    print('\nFunction: f(x,y) = (1 - x)^2 - 100*(y - x^2)^2')

    # Initialization
    x0 = np.array([-1, 2], dtype='float64')
    print('\nStarting point:\nx0 =', x0, '\nf(x0) =', rosen(x0))

    # Run the root search with numerical derivatives
    globvar.steps = 0  # Redundant, but more clear this way
    mini = minimize.qnewton(rosen, grad_rosen, x0, alpha, eps)
    print('\nSolution:\nx =', mini, '\nf(x) =', rosen(mini),
          '\nf\'(x)', grad_rosen(mini))
    print('Steps used: ', globvar.steps)

    # Comparison
    print('\n* Comparison with other methods...')
    globvar.steps = 0
    mini2 = minimize.newton(rosen, grad_rosen, hes_rosen, x0, alpha, eps)
    print('Steps used (Newtons min.): ', globvar.steps)
    globvar.steps = 0
    mini3 = root.newton(grad_rosen, x0, [1e-9, 1e-9], eps)
    print('Steps used (Root, numeric derivs.): ', globvar.steps)
    globvar.steps = 0
    mini4 = root.newton_deriv(grad_rosen, x0, hes_rosen, eps)
    print('Steps used (Root, analytic derivs.): ', globvar.steps)


    ##
    ## PART TWO  -->  Minimization of the Himmelblau function by running
    ##                Quasi Newton's method
    print('\n\n -- Minimization of the Himmelblau function --')
    print('\nFunction: f(x,y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2')

    # Initialization
    x0 = np.array([2, 5], dtype='float64')
    print('\nStarting point:\nx0 =', x0, '\nf(x0) =', himmel(x0))

    # Run the root search with numerical derivatives
    globvar.steps = 0  # Redundant, but more clear this way
    mini = minimize.qnewton(himmel, grad_himmel, x0, alpha, eps)
    print('\nSolution:\nx =', mini, '\nf(x) =', himmel(mini),
          '\nf\'(x)', grad_himmel(mini))
    print('Steps used: ', globvar.steps)

    # Comparison
    print('\n* Comparison with other methods...')
    globvar.steps = 0
    mini2 = minimize.newton(himmel, grad_himmel, hes_himmel, x0, alpha, eps)
    print('Steps used (Newtons min.): ', globvar.steps)
    globvar.steps = 0
    mini3 = root.newton(grad_himmel, x0, [1e-9, 1e-9], eps)
    print('Steps used (Root, numeric derivs.): ', globvar.steps)
    globvar.steps = 0
    mini4 = root.newton_deriv(grad_himmel, x0, hes_himmel, eps)
    print('Steps used (Root, analytic derivs.): ', globvar.steps)
