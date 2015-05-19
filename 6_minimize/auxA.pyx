# cython: language_level=3

# General modules
import numpy as np
import math
import sys

# QR-decomposition routines using Given's rotation
import givens as qr

# Global variables
import globvar


#
# Function to do the Newtons method with known derivatives and backtracking linesearch
#
def newton_min(f, grad, hessian, x0, alpha, eps):
    """
    Finds the minimum of a function using Newton's method with simple back-
    tracking linesearch. The derivatives, in the form of the gradient and the
    Hessian matrix, is supplied by the user. Uses Given's rotation for QR-
    decomposition.

    Returns the approximation of the minimum.

    Side-effect: Changes the global variable 'steps' to the number of steps used

    Arguments:
    - `f`: Function f(x) to find the root of. Argument x is a vector.
    - `grad`: Function which calculates the gradient of f
    - `hessian`: Function which calculates the Hessian matrix of f
    - `x0`: Starting point as a vector
    - `alpha`: Scaling factor of the Armijo condition (for the backtracking)
    - `eps`: Desired accuracy 
    """
    # Initializations
    globvar.steps = 0
    x = np.copy(x0)
    n = len(x)

    # Initial evaluation of the gradient (updates done in bottom of loop)
    df = grad(x)

    # BEGIN MIN SEARCH  -->  Keep going until accuracy reached
    while True:

        # Increase counter and evaluate function
        globvar.steps += 1
        fx = f(x)

        # Evaluate Hessian (gradient is updates in previous iteration)
        H = hessian(x)

        # Decompose and solve using Given's rotation
        qr.decomp(H)
        Dx = -df
        qr.solve(H, Dx)

        # BEGIN BACKTRACKING LINESEARCH
        dot = np.dot(df, Dx)  # For the Armijo condition
        lamb = 2.0  # lambda is a special keyword
        
        while True:
            lamb /= 2
            y = x + Dx*lamb
            fy = f(y)

            # Condition to end  --> The Armijo condition
            if (fy < fx + alpha*lamb*dot) or (lamb < 1/128):
                break
        # END BACKTRACK

        # Store latest approximation
        x = y
        fx = fy
        df = grad(x)

        # Condition to end (derivative vanishes)
        if np.sqrt(np.dot(df, df)) < eps:
            break
        
    # END MIN SEARCH


