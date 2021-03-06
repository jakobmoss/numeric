# cython: language_level=3

# General modules
import numpy as np
import math
import auxA as adapt
import sys

# Library routines for integration
from scipy import integrate

# Global variable to count number of function calls
import globvar

#
# Test-functions to integrate
#
def f1(x):
    globvar.calls += 1
    return math.exp(-math.pow(x, 2)) * 2.0/math.sqrt(math.pi)

def f2(x):
    globvar.calls += 1
    return 1 / (2 + math.sin(x))

def f3(x):
    globvar.calls += 1
    return math.pow(x, 2) / math.exp(x)

def f4(x):
    globvar.calls +=1
    return 4 * math.sqrt(1 - math.pow(1-x, 2))

#
# Main function
#
def mainA():
    """
    Test of the routines for adaptive integration
    """
    ##############
    # Function 1 #
    ##############
    # Conditions
    acc = 1e-3
    eps = 1e-3
    a1 = 0
    b1 = 1
    exact1 = math.erf(1)
    
    # Pretty print!
    print('\n ** Integrating 2*exp(-x^2)/sqrt(pi) from', a1, 'to', b1, ' ** ')
    print('Exact solution = erf(1) ~ {0:.15f}'.format(exact1))

    # Closed quadratures
    globvar.calls = 0
    Qc1, errc1 = adapt.rcquad(f1, a1, b1, acc, eps)
    print('- Integration with closed quadratures:')
    print('Integrand calls =', globvar.calls)
    print('Integral        = {0:.15f}'.format(Qc1))
    print('Error estimate  = {0:.5e}'.format(errc1))
    print('Actual error    = {0:.5e}'.format(abs(Qc1 -  exact1)))

    # Open quadratures
    globvar.calls = 0
    Qo1, erro1 = adapt.roquad(f1, a1, b1, acc, eps)
    print('- Integration with open quadratures:')
    print('Integrand calls =', globvar.calls)
    print('Integral        = {0:.15f}'.format(Qo1))
    print('Error estimate  = {0:.5e}'.format(erro1))
    print('Actual error    = {0:.5e}'.format(abs(Qo1 -  exact1)))


    ##############
    # Function 2 #
    ##############
    # Conditions
    acc = 1e-6
    eps = 1e-6
    a2 = 0
    b2 = 2*math.pi
    exact2 = 2*math.pi/math.sqrt(3)
    
    # Pretty print!
    print('\n ** Integrating 1/(2+sin(x)) from', a2, 'to 2*pi ** ')
    print('Exact solution = 2*pi/sqrt(3) ~ {0:.15f}'.format(exact2))

    # Closed quadratures
    globvar.calls = 0
    Qc2, errc2 = adapt.rcquad(f2, a2, b2, acc, eps)
    print('- Integration with closed quadratures:')
    print('Integrand calls =', globvar.calls)
    print('Integral        = {0:.15f}'.format(Qc2))
    print('Error estimate  = {0:.5e}'.format(errc2))
    print('Actual error    = {0:.5e}'.format(abs(Qc2 -  exact2)))

    # Open quadratures  -->  Here the special flag 'fixzero' is set to prevent
    #                        the integration from exit after just 4 calls to the
    #                        integrand, due to an error estimate of 0 (actual
    #                        error of 2.38977e-01).
    globvar.calls = 0
    Qo2, erro2 = adapt.roquad(f2, a2, b2, acc, eps, fixzero=True)
    print('- Integration with open quadratures:')
    print('Integrand calls =', globvar.calls)
    print('Integral        = {0:.15f}'.format(Qo2))
    print('Error estimate  = {0:.5e}'.format(erro2))
    print('Actual error    = {0:.5e}'.format(abs(Qo2 -  exact2)))


    ##############
    # Function 3 #
    ##############
    # Conditions
    acc = 1e-4
    eps = 1e-4
    a3 = 0
    b3 = 2
    exact3 = 2 - 10/math.exp(2)
    
    # Pretty print!
    print('\n ** Integrating x^2 / exp(x) from', a3, 'to', b3, ' ** ')
    print('Exact solution = 2 - 10/e^2 ~ {0:.15f}'.format(exact3))

    # Closed quadratures
    globvar.calls = 0
    Qc3, errc3 = adapt.rcquad(f3, a3, b3, acc, eps)
    print('- Integration with closed quadratures:')
    print('Integrand calls =', globvar.calls)
    print('Integral        = {0:.15f}'.format(Qc3))
    print('Error estimate  = {0:.5e}'.format(errc3))
    print('Actual error    = {0:.5e}'.format(abs(Qc3 -  exact3)))

    # Open quadratures
    globvar.calls = 0
    Qo3, erro3 = adapt.roquad(f3, a3, b3, acc, eps)
    print('- Integration with open quadratures:')
    print('Integrand calls =', globvar.calls)
    print('Integral        = {0:.15f}'.format(Qo3))
    print('Error estimate  = {0:.5e}'.format(erro3))
    print('Actual error    = {0:.5e}'.format(abs(Qo3 -  exact3)))

    
    ##############
    # Function 4 #
    ##############
    # Conditions
    acc = 1e-12
    eps = 1e-12
    a4= 0
    b4 = 1
    exact4 = math.pi
    
    # Pretty print!
    print('\n ** Integrating 4*sqrt(1-(1-x^2)) from', a4, 'to', b4, ' ** ')
    print('Exact solution = pi ~ {0:.16f}'.format(exact4))

    # Closed quadratures
    globvar.calls = 0
    Qc4, errc4 = adapt.rcquad(f4, a4, b4, acc, eps)
    print('- Integration with closed quadratures:')
    print('Integrand calls =', globvar.calls)
    print('Integral        = {0:.16f}'.format(Qc4))
    print('Error estimate  = {0:.9e}'.format(errc4))
    print('Actual error    = {0:.9e}'.format(abs(Qc4 -  exact4)))

    # Open quadratures
    globvar.calls = 0
    Qo4, erro4 = adapt.roquad(f4, a4, b4, acc, eps)
    print('- Integration with open quadratures:')
    print('Integrand calls =', globvar.calls)
    print('Integral        = {0:.16f}'.format(Qo4))
    print('Error estimate  = {0:.9e}'.format(erro4))
    print('Actual error    = {0:.9e}'.format(abs(Qo4 -  exact4)))

    # SciPy integration routine (from QUADPACK)
    globvar.calls = 0
    Qsp, errsp = integrate.quad(f4, a4, b4, epsabs=acc, epsrel=eps)
    print('- Integration with SciPy routine (from QUADPACK):')
    print('Integrand calls =', globvar.calls)
    print('Integral        = {0:.16f}'.format(Qsp))
    print('Error estimate  = {0:.9e}'.format(errsp))
    print('Actual error    = {0:.9e}'.format(abs(Qsp -  exact4)))
