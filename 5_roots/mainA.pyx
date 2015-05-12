# cython: language_level=3

import numpy as np
import auxA as root

#
# The Rosenbrock valley function (or actually the gradient of it)
#
def rosen(q):
    # Global variable to count the number of calls to the function
    global ncalls
    ncalls += 1
        
    # Init
    z = np.zeros(2, dtype='float64')
    x = q[0]
    y = q[1]

    # Calculate and return
    z[0] = 2*(1-x)*(-1) + 100*2*(y-x*x)*(-1)*2*x 
    z[1] = 100*2*(y-x*x)
    return z


#
# Main function
#
def mainA():
    """
    Test of the root finding methods
    """
    # Global variable to count the number of function calls

    ## PART TWO  -->  Root search of the Rosenbrock valley function by running
    ##                Newton's method on the gradient
    print('\n -- Root search of the Rosenbrock valley function --')

    # Initialization
    global ncalls
    ncalls = 0
    x0 = np.array([5, 10], dtype='float64')
    dx = np.array([1e-7, 1e-7], dtype='float64')
    print('\nStarting point:\nx0 =', x0, '\nf(x0) =', rosen(x0))

    # Run the root search
    roots = root.newton(rosen, x0, dx, 1e-6)
    print('\nSolution:\nx =', roots, '\nf(x) =', rosen(roots))
    print('\nNumber of calls to the function:\nn =', ncalls)
    
    
