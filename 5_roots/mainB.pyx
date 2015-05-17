# cython: language_level=3

import numpy as np
import auxB as root
import math, sys

# Global variables
import globvar

# Different functions to root-find and their derivatives
from datafun import *


#
# Main function
#
def mainB():
    """
    Test of the root finding methods
    """
    ## PART ZERO  --> Pretty print!
    print('\n*****************************************')
    print('* Comparison of Newton\'s method with    *')
    print('*  - Numerical estimate of the Jacobian *')
    print('*  - User-supplied Jacobian             *')
    print('*****************************************\n')
    
    ## PART ONE  -->  Solving a system of equations by root search with
    ##                Newton's method
    print('\n -- Solve system of equations by root search --')
    print('\nSystem: Axy=1 ;  exp(-x)+exp(-y)=1 + 1/A')

    # Initialization
    x0 = np.array([3, -1], dtype='float64')
    print('\nStarting point:\nx0 =', x0, '\nf(x0) =', eqsys(x0))

    # Run the root search with numerical derivatives
    globvar.ncalls = 0
    dx = np.array([1e-9, 1e-9], dtype='float64')
    roots1 = root.newton(eqsys, x0, dx, 1e-12)
    print('\nSolution (numerical):\nx =', roots1, '\nf(x) =', eqsys(roots1))
    print('Calls to the function: ', globvar.ncalls)

    # Run the root search with user-supplied derivatives
    globvar.ncalls = 0
    roots2 = root.newton_deriv(eqsys, x0, diff_eqsys, 1e-12)
    print('\nSolution (user-sup):\nx =', roots2, '\nf(x) =', eqsys(roots2))
    print('Calls to the function: ', globvar.ncalls)

    
    ## PART TWO  -->  Root search of the Rosenbrock valley function by running
    ##                Newton's method on the gradient
    print('\n\n -- Root search of the Rosenbrock valley function --')
    print('\nFunction: f(x,y) = (1 - x)^2 - 100*(y - x^2)^2')

    # Initialization
    globvar.ncalls = 0
    x0 = np.array([-1, 5], dtype='float64')
    dx = np.array([1e-7, 1e-7], dtype='float64')
    print('\nStarting point:\nx0 =', x0, '\nf(x0) =', rosen(x0))

    # Run the root search
    roots2 = root.newton(rosen, x0, dx, 1e-8)
    print('\nSolution:\nx =', roots2, '\nf(x) =', rosen(roots2))
    print('\nNumber of calls to the function:\nn =', globvar.ncalls)
    
    
    ## PART THREE  -->  Root search of the Himmelblau function by running
    ##                Newton's method on the gradient
    print('\n\n -- Root search of the Himmelblau function --')
    print('\nFunction: f(x,y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2')

    # Initialization
    globvar.ncalls = 0
    x0 = np.array([1, -1], dtype='float64')
    dx = np.array([1e-9, 1e-9], dtype='float64')
    print('\nStarting point:\nx0 =', x0, '\nf(x0) =', himmel(x0))

    # Run the root search
    roots3 = root.newton(himmel, x0, dx, 1e-12)
    print('\nSolution:\nx =', roots3, '\nf(x) =', himmel(roots3))
    print('\nNumber of calls to the function:\nn =', globvar.ncalls)


    ## PART FOUR  -->  Root search of the Matyas function by running
    ##                Newton's method on the gradient
    print('\n\n -- Root search of the Matyas function --')
    print('\nFunction: f(x,y) = 0.26*(x^2 + y^2) - 0.48*xy')

    # Initialization
    globvar.ncalls = 0
    x0 = np.array([2, -7], dtype='float64')
    dx = np.array([1e-9, 1e-9], dtype='float64')
    print('\nStarting point:\nx0 =', x0, '\nf(x0) =', matya(x0))

    # Run the root search
    roots4 = root.newton(matya, x0, dx, 1e-12)
    print('\nSolution:\nx =', roots4, '\nf(x) =', matya(roots4))
    print('\nNumber of calls to the function:\nn =', globvar.ncalls)
