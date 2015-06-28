# cython: language_level=3

# General modules
import numpy as np
import math
import sys

#
# CLOSED QUAD  -->  Interface
#
def rcquad(F, a, b, acc, eps):
    """
    Recursive adaptive integration using Closed QUADratures. Uses a 4th order
    trapezium rule for evaluation with 2nd order rectangular rule for error 
    estimation. Re-uses points to minimize number of evaluations.

    Returns the integral of F(x) from a to b, and an estimate of the error.
    
    Arguments:
    - `F`: Function to integrate
    - `a`: Stating point
    - `b`: End point
    - `acc`: Allowed absolute error
    - `eps`: Allowed relative error
    """
    # If the limits are okay: Perform the integration
    if a < b:
        # Pre-calculate the end-points (re-usable)
        h = b - a
        x1 = 0; x4 = 1
        f1 = F(a + h*x1)
        f4 = F(a + h*x4)

        # Call the integrator and return
        Q, err = rcquad24(F, a, b, f1, f4, acc, eps, 0)
        return Q, err
    else:
        print('Check your limits!', file=sys.stderr)
        
#
# CLOSED QUAD  -->  Integrator
#
def rcquad24(F, a, b, f1, f4, acc, eps, nrec):
    """
    The actual integrator. Is called from the interface function rcquad.

    Integrates the function F from a to b, re-using previously estimated 
    points. If the error is to big, the interval is sub-divided and the 
    integrator calls itself (hence, recursive).
    
    Specific arguments (re-usable points):
    - `f1`: Estimated value of F(a)
    - `f4`: Estimated value of F(b)
    - `nrec`: The current level of recursion
    """
    # Report an error if the recursion gets to deep
    if nrec > 15000:
        print('\nrcquad24: To many recursions!', file=sys.stderr)
        return 0

    # Estimate of the two remaining points
    h = b - a
    x2 = 1/3.0; x3 = 2/3.0
    f2 = F(a + h*x2)
    f3 = F(a + h*x3)

    # Weights of the 4th order trapezium rule
    w1 = 1/8.0; w2 = 3/8.0; w3 = 3/8.0; w4 = 1/8.0

    # Weights of the 2nd order rectangular rule
    v1 = 1/4.0; v2 = 1/4.0; v3 = 1/4.0; v4 = 1/4.0

    # Calculate the quadratures of higher order Q and lower order q
    Q = h * (w1*f1 + w2*f2 + w3*f3 + w4*f4)
    q = h * (v1*f1 + v2*f2 + v3*f3 + v4*f4)

    # Estimate of the local error and tolerance
    err = abs(Q - q)
    tol = acc + eps*abs(Q)

    # If the error is small: Accept the integration; return estimate and error
    if err < tol:
        return Q, err
    #
    # If not: Split the interval in three and integrate each seperately with a
    # scaled absolute accuracy goal. Appropriate points are re-used.
    else:
        accscale = math.sqrt(3)
        Ql, errl = rcquad24(F, a, a+h/3.0, f1, f2, acc/accscale, eps, nrec+1)
        Qm, errm = rcquad24(F, a+h/3.0, a+2*h/3.0, f2, f3, acc/accscale, eps, nrec+1)
        Qr, errr = rcquad24(F, a+2*h/3.0, b, f3, f4, acc/accscale, eps, nrec+1)
        Qtot = Ql + Qm + Qr
        errtot = math.sqrt(errl*errl + errm*errm + errr*errr)
        return Qtot, errtot


    
#
# OPEN QUAD  -->  Interface
#
def roquad(F, a, b, acc, eps):
    """
    Recursive adaptive integration using Open QUADratures. Uses a 4th order
    trapezium rule for evaluation with 2nd order rectangular rule for error 
    estimation. Re-uses points to minimize number of evaluations.

    Returns the integral of F(x) from a to b, and an estimate of the error.
    
    Arguments:
    - `F`: Function to integrate
    - `a`: Stating point
    - `b`: End point
    - `acc`: Allowed absolute error
    - `eps`: Allowed relative error
    """
    # If the limits are okay: Perform the integration
    if a < b:
        # Pre-calculate the end-points (re-usable)
        h = b - a
        x2 = 1/3.0; x3 = 2/3.0  # From the points 2/6 and 4/6
        f2 = F(a + h*x2)
        f3 = F(a + h*x3)

        # Call the integrator and return
        Q, err = roquad24(F, a, b, f2, f3, acc, eps, 0)
        return Q, err
    else:
        print('Check your limits!', file=sys.stderr)
        
#
# OPEN QUAD  -->  Integrator
#
def roquad24(F, a, b, f2, f3, acc, eps, nrec):
    """
    The actual integrator. Is called from the interface function rcquad.

    Integrates the function F from a to b, re-using previously estimated 
    points. If the error is to big, the interval is sub-divided and the 
    integrator calls itself (hence, recursive).
    
    Specific arguments (re-usable points):
    - `f2`: Estimated value at x2
    - `f3`: Estimated value at x3
    - `nrec`: The current level of recursion
    """
    # Report an error if the recursion gets to deep
    if nrec > 15000:
        print('\nroquad24: To many recursions!', file=sys.stderr)
        return 0

    # Estimate of the two remaining points
    h = b - a
    x1 = 1/6.0; x4 = 5/6.0
    f1 = F(a + h*x1)
    f4 = F(a + h*x4)

    # Weights of the 4th order trapezium rule
    w1 = 2/6.0; w2 = 1/6.0; w3 = 1/6.0; w4 = 2/6.0

    # Weights of the 2nd order rectangular rule
    v1 = 1/4.0; v2 = 1/4.0; v3 = 1/4.0; v4 = 1/4.0

    # Calculate the quadratures of higher order Q and lower order q
    Q = h * (w1*f1 + w2*f2 + w3*f3 + w4*f4)
    q = h * (v1*f1 + v2*f2 + v3*f3 + v4*f4)

    # Estimate of the local error and tolerance
    err = abs(Q - q)
    tol = acc + eps*abs(Q)

    # If the error is small: Accept the integration; return estimate and error
    if err < tol:
        return Q, err
    #
    # If not: Split the interval in two and integrate each seperately with a
    # scaled absolute accuracy goal. Appropriate points are re-used.
    else:
        accscale = math.sqrt(2)
        Ql, errl = roquad24(F, a, a+h/2.0, f1, f2, acc/accscale, eps, nrec+1)
        Qr, errr = roquad24(F, a+h/2.0, b, f3, f4, acc/accscale, eps, nrec+1)
        Qtot = Ql + Qr
        errtot = math.sqrt(errl*errl + errr*errr)
        return Qtot, errtot
