# cython: language_level=3

# General modules
import numpy as np
import math
import sys

# QR-decomposition routines using Given's rotation
import givens as qr


#
# Function to find the root of a function with Newton's method
#
def newton(f, x0, dx, eps):
    """
    Finds the root of a function using Newton's method with simple
    backtracking linesearch and numerical evaluation of the Jacobian. Uses 
    Given's rotation for QR-decomposition.

    Returns the approximation of the root.


    Arguments:
    - `f`: Function f(x) to find the root of. Argument x is a vector.
    - `x0`: Starting point as a vector
    - `dx`: The deltaX to be used in evaluation of the Jacobian [vector]
    - `eps`: Desired accuracy 
    """
    # Initializations
    x = np.copy(x0)
    n = len(x)
    J = np.zeros((n, n), dtype='float64')

    # BEGIN ROOT SEARCH  -->  Keep going until accuracy reached
    while True:
        fx = f(x)

        # BEGIN TO FILL JACOBIAN -> J  -->  Numerical estimate
        for j in range(n):
            x[j] += dx[j]
            df = f(x) - fx

            for i in range(n):
                J[i, j] = df[i] / dx[j]

            x[j] -= dx[j]
        # END JACOBIAN

        # Decompose and solve using Given's rotation
        qr.decomp(J)
        Dx = -fx
        qr.solve(J, Dx)

        # BEGIN BACKTRACKING LINESEARCH
        lamb = 2.0  # lambda is a special keyword
        while True:
            lamb /= 2
            y = x + Dx*lamb
            fy = f(y)

            # Condition to end  --  Note: norm(x) := np.sqrt(np.dot(x,x))
            if (np.sqrt(np.dot(fy, fy)) < (1 - lamb/2)*np.sqrt(np.dot(fx, fx))) \
               or (lamb < 1/128.0):
                break
        # END BACKTRACK

        # Store latest approximation
        x = y
        fx = fy

        # Condition to end
        if (np.sqrt(np.dot(Dx, Dx)) < np.sqrt(np.dot(dx, dx))) \
           or (np.sqrt(np.dot(fx, fx)) < eps):
            break
        
    # END ROOT SEARCH

    # Return the accepted approximation
    return x
