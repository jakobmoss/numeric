# cython: language_level=3

# General modules
import numpy as np


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
    c4 = 0

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
    k1 = F(x + c1*h, y)
    k2 = F(x + c2*h, y + a21*k1)
    k3 = F(x + c3*h, y + a31*k1 + a32*k2)
    k4 = F(x + c4*h, y + a41*k1 + a42*k2 + a43*k3)

    # Approximate next step and error
    yh = y + h*(b1*k1 + b2*k2 + b3*k3 + b4*k4)
    yhs = y + h*(bs1*k1 + bs2*k2 + bs3*k3 + bs4*k4)
    err = yh - yhs

    # Return approximation and error
    return yh, err
