# cython: language_level=3

import numpy as np
import math
import auxA as adapt
import sys

# Global variable to count number of function calls
import globvar

#
# Test-functions to integrate
#
def f1(x):
    globvar.calls += 1
    return math.exp(-x**2) * 2.0/math.sqrt(math.pi)

#
# Main function
#
def mainA():
    """
    Test of the ODE routines
    """
    ##############
    # Function 1 #
    ##############
    # Conditions
    acc = 1e-3
    eps = 1e-3
    a1 = 0
    b1 = 1
    
    # Pretty print!
    print('\n ** Integrating 2*exp(-x^2)/sqrt(pi) from', a1, 'to', b1, ' ** ')
    print('Exact solution = erf(1) ~ {0:.15f}'.format(math.erf(1)))

    # Closed quadratures
    globvar.calls = 0
    Qc1, errc1 = adapt.rcquad(f1, a1, b1, acc, eps)
    print('- Integration with closed quadratures:')
    print('Integrand calls =', globvar.calls)
    print('Integral        = {0:.15f}'.format(Qc1))
    print('Error estimate  = {0:.5e}'.format(errc1))
    print('Actual error    = {0:.5e}'.format(abs(Qc1 -  0.84270079295)))

    # Open quadratures
    globvar.calls = 0
    Qo1, erro1 = adapt.roquad(f1, a1, b1, acc, eps)
    print('- Integration with open quadratures:')
    print('Integrand calls =', globvar.calls)
    print('Integral        = {0:.15f}'.format(Qo1))
    print('Error estimate  = {0:.5e}'.format(erro1))
    print('Actual error    = {0:.5e}'.format(abs(Qo1 -  0.84270079295)))
