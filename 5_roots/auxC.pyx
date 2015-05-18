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
               or (lamb < 1/128):
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


#
# Newton with better linesearch
#
def newton_quad(f, x0, dx, eps):
    """
    Finds the root of a function using Newton's method with quadratic
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
        nfx = np.sqrt(np.dot(fx, fx))   # The norm (used multiple times)

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

        # BEGIN BACKTRACKING LINESEARCH  --> QUADRATIC APPROXIMATION

        # Initial attempt
        lamb = 1.0  # lambda is a special keyword
        y = x + Dx*lamb
        fy = f(y)
        nfy = np.sqrt(np.dot(fy, fy))

        # Known values of the function to minimize [6.9]
        g0 = 0.5*nfx*nfx  # g(0)
        gp0 = (-1)*nfx*nfx    # g'(0)

        # Only enter if the step is rejected
        while (nfy > (1 - lamb/2)*nfx) and (lamb > 1.0/128):

            # Calculate approximation [6.11]
            gl = 0.5*nfy*nfy  # g(\lambda)
            c = (gl - g0 - gp0*lamb)/(lamb*lamb)

            # Update step
            lamb = -gp0/(2*c)
            y = x + Dx*lamb
            fy = f(y)
            nfy = np.sqrt(np.dot(fy, fy))
            
        # END BACKTRACK

        # Store latest approximation
        x = y
        fx = fy
        nfx = np.sqrt(np.dot(fx, fx))

        # Condition to end
        if (np.sqrt(np.dot(Dx, Dx)) < np.sqrt(np.dot(dx, dx))) \
           or (nfx < eps):
            break
        
    # END ROOT SEARCH

    # Return the accepted approximation
    return x
