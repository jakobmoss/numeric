# cython: language_level=3

# General modules
import numpy as np
import math
import sys


#
# Embedded Runge-Kutta stepper
#
def rkstep23(F, x, y, h):
    """
    Embedded Runge-Kutta stepper of orders 3 and 2, using the Bogacki-Shampine
    method.

    Returns etimates of function and error on the step

    Arguments:
    - `F`: Function containing the right-hand-side
    - `x`: Current location
    - `y`: Current value of the function
    - `h`: Step-size
    """
    # BEGIN  -->  BUTCHER'S TABLEAU
    # The nodes, c
    c1 = 0
    c2 = 1/2.0
    c3 = 3/4.0
    c4 = 1

    # The Runge-Kutta matrix
    a21 = 1/2.0
    a31 = 0
    a32 = 3/4.0
    a41 = 2/9.0
    a42 = 1/3.0
    a43 = 4/9.0

    # The weights, b
    b1 = 2/9.0
    b2 = 1/3.0
    b3 = 4/9.0
    b4 = 0

    # The weights, b*
    bs1 = 7/24.0
    bs2 = 1/4.0
    bs3 = 1/3.0
    bs4 = 1/8.0
    # END  -->  BUTCHER'S TABLEAU

    # Calculate the k's
    k1 = h*F(x + c1*h, y)
    k2 = h*F(x + c2*h, y + a21*k1)
    k3 = h*F(x + c3*h, y + a31*k1 + a32*k2)
    k4 = h*F(x + c4*h, y + a41*k1 + a42*k2 + a43*k3)

    # Approximate next step and error
    yh = y + b1*k1 + b2*k2 + b3*k3 + b4*k4
    yhs = y + bs1*k1 + bs2*k2 + bs3*k3 + bs4*k4
    err = yh - yhs
    normerr = np.sqrt(np.dot(err, err))
    
    # Return approximation and error
    return yh, normerr


#
# Another simple stepper for testing purposes
#
def rkstep12(F, x, y, h):
    """
    Embedded Runge-Kutta stepper of orders 2 and 1, using the mid-point
    method.

    Returns etimates of function and error on the step

    Arguments:
    - `F`: Function containing the right-hand-side
    - `x`: Current location
    - `y`: Current value of the function
    - `h`: Step-size
    """
    # Coefficients
    k0  = F(x, y)
    k12 = F(x + 0.5*h, y + 0.5*h*k0)

    # Approximation of function and step-error
    yh = y + k12*h
    err = (k0 - k12)*0.5*h
    normerr = np.sqrt(np.dot(err, err))

    # Return
    return yh, normerr


#
# Driver for solving ODE
#
def rkdriver(F, a, b, ya, h, acc, eps, method):
    """
    Evolves a function from a to b using a specified Runge-Kutta stepper and
    adaptive step-size

    Returns the list of steps and calculated function values

    Arguments:
    - `F`: Function to evolve
    - `a`: Starting point
    - `b`: End point
    - `ya`: Function value at starting point
    - `h`: Initial step-size
    - `acc`: Absolute precision
    - `eps`: Relative presision
    - `method`: Which stepper to use (as a string)
    """
    # Which stepper to use
    if method.lower() in ['rkstep23', 'rk23']:
        stepper = rkstep23
    elif method.lower() in ['rkstep12', 'rk12']:
        stepper = rkstep12
    else:
        print('Unknown stepper selected!', file=sys.stderr)
        return
        
    # Initializations
    power = 0.25
    safety = 0.95
    xs = np.array([a], dtype='float')
    ys = np.array([ya], dtype='float')

    # BEGIN  -->  EVOLVE TOWARDS B
    while True:
        x = xs[-1]
        y = ys[-1]

        # CONDITION TO END: Have we reached the end of the interval?
        if x >= b:
            break

        # If the step would end outside the interval: Step to the edge
        if (x+h) > b:
            h = b - x

        # Perform the step
        yh, err = stepper(F, x, y, h)

        # Calculate local tolerance -- Note: norm(x) := np.sqrt(np.dot(x,x))
        tol = (eps*np.sqrt(np.dot(yh, yh)) + acc) * math.sqrt(h/(b-a))

        # If local error less than the local tolerance: Accept the step
        if err < tol:
            xs = np.append(xs, x+h)
            ys = np.vstack([ys, yh])

        # If the error is non-zero: decrese the step. Otherwise: double it.
        if err > 0:
            h *= math.pow(tol/err, power) * safety
        else:
            h *= 2

    # END EVOLUTION

    # Return the stored data
    return xs, ys
