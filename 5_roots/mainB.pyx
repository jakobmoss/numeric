# cython: language_level=3

import numpy as np
import auxB as root
import math, sys


#
# Different functions to root-find
#
def eqsys(q):
    """
    System of equations (A = 10000):
    A x y = 1
    exp(-x) + exp(-y) = 1 + 1/A
    """
    # Global variable to count the number of calls to the function
    global ncalls
    ncalls += 1

    # Init
    z = np.zeros(2, dtype='float64')
    x = q[0]
    y = q[1]
    A = 10000.0

    # Calculate and return
    z[0] = A*x*y - 1
    z[1] = math.exp(-x) + math.exp(-y) - 1 - 1/A
    return z
    

def rosen(q):
    """
    The Rosenbrock valley function (or actually the gradient of it)
    """
    # Global variable to count the number of calls to the function
    global ncalls
    ncalls += 1
        
    # Init
    z = np.zeros(2, dtype='float64')
    x = q[0]
    y = q[1]

    # Calculate and return
    z[0] = 2*(x-1) - 400*(y-x*x)*x 
    z[1] = 200*(y-x*x)
    return z


def himmel(q):
    """
    The Himmelblau function (or actually the gradient of it)
    """
    # Global variable to count the number of calls to the function
    global ncalls
    ncalls += 1
        
    # Init
    z = np.zeros(2, dtype='float64')
    x = q[0]
    y = q[1]

    # Calculate and return
    z[0] = 4*(x*x+y-11)*x + 2*(x+y*y-7)
    z[1] = 2*(x*x+y-11) + 4*(x+y*y-7)*y
    return z


def matya(q):
    """
    The Matyas function (or actually the gradient of it)
    """
    # Global variable to count the number of calls to the function
    global ncalls
    ncalls += 1
        
    # Init
    z = np.zeros(2, dtype='float64')
    x = q[0]
    y = q[1]

    # Calculate and return
    z[0] = 2*0.26*x - 0.48*y
    z[1] = 2*0.26*y - 0.48*x
    return z


#
# Main function
#
def mainB():
    """
    Test of the root finding methods
    """
    # Global variable to count calls to the functions
    global ncalls

    ## PART ONE  -->  Solving a system of equations by root search with
    ##                Newton's method
    print('\n -- Solve system of equations by root search --')
    print('\nSystem: Axy=1 ;  exp(-x)+exp(-y)=1 + 1/A')

    # Initialization
    ncalls = 0
    x0 = np.array([3, -1], dtype='float64')
    dx = np.array([1e-9, 1e-9], dtype='float64')
    print('\nStarting point:\nx0 =', x0, '\nf(x0) =', eqsys(x0))

    # Run the root search
    roots1 = root.newton(eqsys, x0, dx, 1e-12)
    print('\nSolution:\nx =', roots1, '\nf(x) =', eqsys(roots1))
    print('\nNumber of calls to the function:\nn =', ncalls)

    
    ## PART TWO  -->  Root search of the Rosenbrock valley function by running
    ##                Newton's method on the gradient
    print('\n\n -- Root search of the Rosenbrock valley function --')
    print('\nFunction: f(x,y) = (1 - x)^2 - 100*(y - x^2)^2')

    # Initialization
    ncalls = 0
    x0 = np.array([-1, 5], dtype='float64')
    dx = np.array([1e-7, 1e-7], dtype='float64')
    print('\nStarting point:\nx0 =', x0, '\nf(x0) =', rosen(x0))

    # Run the root search
    roots2 = root.newton(rosen, x0, dx, 1e-8)
    print('\nSolution:\nx =', roots2, '\nf(x) =', rosen(roots2))
    print('\nNumber of calls to the function:\nn =', ncalls)
    
    
    ## PART THREE  -->  Root search of the Himmelblau function by running
    ##                Newton's method on the gradient
    print('\n\n -- Root search of the Himmelblau function --')
    print('\nFunction: f(x,y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2')

    # Initialization
    ncalls = 0
    x0 = np.array([1, -1], dtype='float64')
    dx = np.array([1e-9, 1e-9], dtype='float64')
    print('\nStarting point:\nx0 =', x0, '\nf(x0) =', himmel(x0))

    # Run the root search
    roots3 = root.newton(himmel, x0, dx, 1e-12)
    print('\nSolution:\nx =', roots3, '\nf(x) =', himmel(roots3))
    print('\nNumber of calls to the function:\nn =', ncalls)


    ## PART FOUR  -->  Root search of the Matyas function by running
    ##                Newton's method on the gradient
    print('\n\n -- Root search of the Matyas function --')
    print('\nFunction: f(x,y) = 0.26*(x^2 + y^2) - 0.48*xy')

    # Initialization
    ncalls = 0
    x0 = np.array([2, -7], dtype='float64')
    dx = np.array([1e-9, 1e-9], dtype='float64')
    print('\nStarting point:\nx0 =', x0, '\nf(x0) =', matya(x0))

    # Run the root search
    roots4 = root.newton(matya, x0, dx, 1e-12)
    print('\nSolution:\nx =', roots4, '\nf(x) =', matya(roots4))
    print('\nNumber of calls to the function:\nn =', ncalls)
